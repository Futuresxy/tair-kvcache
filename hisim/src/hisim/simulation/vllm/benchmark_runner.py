"""HiSim benchmark-runner interface backed by the vLLM 0.23 scheduler."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from hisim.dataset import BaseDataset, DatasetArgs, get_dataset
from hisim.simulation.base.runner import BaseBenchmarkRunner
from hisim.simulation.types import BenchmarkConfig, RequestStats
from hisim.simulation.utils import calc_metrics

from .scheduler_simulator import (
    LinearLatencyProfile,
    SimulationResult,
    VllmRequestSpec,
    VllmSchedulerSimulator,
)
from .simulate import load_latency_profile


class VllmBenchmarkRunner(BaseBenchmarkRunner):
    """Expose the vLLM scheduler simulator through HiSim's runner contract.

    One runner is one ``instance_id`` and therefore one KV-cache namespace.
    Cache contents persist between ``benchmark`` calls until ``flush_cache`` is
    invoked, matching the cold/warm workflow of ``SGLangBenchmarkRunner``.
    """

    def __init__(
        self,
        *,
        model: str,
        instance_id: str,
        latency_profile: LinearLatencyProfile | None = None,
        latency_profile_path: str | Path | None = None,
        block_size: int = 16,
        num_gpu_blocks: int = 4096,
        max_model_len: int = 4096,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 256,
        random_seed: int = 0,
    ) -> None:
        if latency_profile is not None and latency_profile_path is not None:
            raise ValueError(
                "latency_profile and latency_profile_path are mutually exclusive"
            )
        if latency_profile_path is not None:
            latency_profile = load_latency_profile(Path(latency_profile_path))
        self.model = model
        self.instance_id = instance_id
        self.random_seed = random_seed
        self.simulator = VllmSchedulerSimulator(
            model=model,
            instance_id=instance_id,
            latency_profile=latency_profile,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        self._tokenizer = None
        self._run_index = 0
        self._last_result: SimulationResult | None = None
        self._last_request_stats: list[dict] = []

    def benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: Optional[BaseDataset] = None,
        dataset_args: Optional[DatasetArgs] = None,
    ) -> dict:
        if (dataset is None) == (dataset_args is None):
            raise ValueError(
                "Exactly one of `dataset` or `dataset_args` must be provided."
            )
        if dataset is None:
            from transformers import AutoTokenizer

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model, local_files_only=True, trust_remote_code=False
                )
            dataset = get_dataset(dataset_args, tokenizer=self._tokenizer)

        workload = self._make_workload(dataset, benchmark_config)
        result = self.simulator.run(workload)
        self._last_result = result
        self._last_request_stats = [request.to_dict() for request in result.requests]
        self._run_index += 1

        request_stats = [self._to_hisim_request_stats(request) for request in result.requests]
        metrics = calc_metrics(request_stats)
        metrics.update(
            {
                "backend": "vllm",
                "backend_version": result.vllm_version,
                "instance_id": result.instance_id,
                "prefix_cache_queries": result.prefix_cache_queries,
                "prefix_cache_hits": result.prefix_cache_hits,
                "kv_cache_hit_rate": result.kv_cache_hit_rate,
                "latency_profile": asdict(result.latency_profile),
            }
        )
        return metrics

    def _make_workload(
        self, dataset: BaseDataset, benchmark_config: BenchmarkConfig
    ) -> list[VllmRequestSpec]:
        rng = np.random.default_rng(self.random_seed + self._run_index)
        next_arrival_ms = 0.0
        requests = []
        for index, request in enumerate(dataset):
            token_ids = request.token_ids
            if token_ids is None:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model, local_files_only=True, trust_remote_code=False
                    )
                token_ids = self._tokenizer.encode(
                    request.prompt, add_special_tokens=False
                )

            if benchmark_config.ignore_request_timestamp:
                arrival_time_ms = next_arrival_ms
                if not math.isinf(benchmark_config.request_rate):
                    if benchmark_config.request_rate <= 0:
                        raise ValueError("request_rate must be positive")
                    next_arrival_ms += (
                        rng.exponential(1.0 / benchmark_config.request_rate) * 1000.0
                    )
            else:
                arrival_time_ms = (
                    float(request.custom_params.get("created_time", 0.0)) * 1000.0
                )
            requests.append(
                VllmRequestSpec(
                    request_id=f"run-{self._run_index}-request-{index}",
                    prompt_token_ids=list(token_ids),
                    max_tokens=request.output_length,
                    instance_id=self.instance_id,
                    arrival_time_ms=arrival_time_ms,
                )
            )
        return requests

    @staticmethod
    def _to_hisim_request_stats(request) -> RequestStats:
        token_latencies_ms = []
        if request.ttft_ms is not None:
            token_latencies_ms.append(request.ttft_ms)
        token_latencies_ms.extend(request.inter_token_latencies_ms)
        return RequestStats(
            rid=request.request_id,
            last_event_time=(request.finish_time_ms or 0.0) / 1000.0,
            input_length=request.prompt_tokens,
            output_length=request.output_tokens,
            final_reused_tokens=request.cached_tokens,
            queue_start=request.arrival_time_ms / 1000.0,
            queue_end=(request.first_scheduled_time_ms or request.arrival_time_ms)
            / 1000.0,
            created_time=request.arrival_time_ms / 1000.0,
            gen_token_latencies=[latency / 1000.0 for latency in token_latencies_ms],
        )

    def get_request_stats(self) -> list[dict]:
        return list(self._last_request_stats)

    def get_iteration_stats(self) -> list[dict]:
        if self._last_result is None:
            return []
        return [asdict(step) for step in self._last_result.steps]

    def flush_cache(self) -> None:
        self.simulator.reset_prefix_cache()

    def shutdown(self) -> None:
        self._last_result = None
