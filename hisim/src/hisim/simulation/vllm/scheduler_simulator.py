"""Drive the real vLLM 0.23 scheduler without executing model kernels.

This module is the first layer of the HiSim/vLLM integration.  vLLM owns
request admission, batching, block allocation, prefix hashing, prefix-cache
lookup, preemption and request completion.  HiSim replaces only the GPU model
runner with a deterministic token producer and a latency model.  Consequently,
cache-hit and scheduling behaviour comes from the selected vLLM version rather
than from a second, approximate scheduler implementation.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


COMPATIBLE_VLLM_VERSIONS = ("0.23.0",)


@dataclass(frozen=True)
class VllmRequestSpec:
    """Token-level workload request consumed by :class:`VllmSchedulerSimulator`.

    ``arrival_time_ms`` is relative to the beginning of one simulation run.
    ``instance_id`` is mandatory because KV blocks must never be reused across
    HiSim instances.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    instance_id: str
    arrival_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be non-negative")


@dataclass(frozen=True)
class LinearLatencyProfile:
    """Simple replaceable latency model used until an RTX 4090 fit is loaded.

    A step can contain both prefill and decode work.  The current model sums the
    two contributions.  Its coefficients are deliberately explicit in result
    files so provisional runs cannot be confused with calibrated measurements.
    """

    name: str = "uncalibrated-linear"
    scheduler_overhead_ms: float = 0.05
    prefill_base_ms: float = 0.30
    prefill_token_ms: float = 0.015
    decode_base_ms: float = 0.20
    decode_token_ms: float = 0.04
    decode_context_token_ms: float = 0.0
    calibrated: bool = False

    def predict_ms(
        self,
        prefill_tokens: int,
        decode_tokens: int,
        decode_context_tokens: int = 0,
    ) -> float:
        latency = self.scheduler_overhead_ms
        if prefill_tokens:
            latency += self.prefill_base_ms + self.prefill_token_ms * prefill_tokens
        if decode_tokens:
            latency += (
                self.decode_base_ms
                + self.decode_token_ms * decode_tokens
                + self.decode_context_token_ms * decode_context_tokens
            )
        return latency


@dataclass
class RequestMetrics:
    request_id: str
    instance_id: str
    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int = 0
    first_scheduled_time_ms: float | None = None
    first_token_time_ms: float | None = None
    finish_time_ms: float | None = None
    token_timestamps_ms: list[float] = field(default_factory=list)
    cached_tokens: int = 0
    local_cached_tokens: int = 0
    external_cached_tokens: int = 0
    computed_prompt_tokens: int = 0
    finish_reason: str | None = None

    @property
    def queue_time_ms(self) -> float | None:
        if self.first_scheduled_time_ms is None:
            return None
        return self.first_scheduled_time_ms - self.arrival_time_ms

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time_ms is None:
            return None
        return self.first_token_time_ms - self.arrival_time_ms

    @property
    def e2e_latency_ms(self) -> float | None:
        if self.finish_time_ms is None:
            return None
        return self.finish_time_ms - self.arrival_time_ms

    @property
    def tpot_ms(self) -> float | None:
        if len(self.token_timestamps_ms) <= 1:
            return None
        return (self.token_timestamps_ms[-1] - self.token_timestamps_ms[0]) / (
            len(self.token_timestamps_ms) - 1
        )

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        return [
            current - previous
            for previous, current in zip(
                self.token_timestamps_ms, self.token_timestamps_ms[1:]
            )
        ]

    @property
    def kv_cache_hit_rate(self) -> float:
        if not self.prompt_tokens:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            queue_time_ms=self.queue_time_ms,
            ttft_ms=self.ttft_ms,
            tpot_ms=self.tpot_ms,
            e2e_latency_ms=self.e2e_latency_ms,
            inter_token_latencies_ms=self.inter_token_latencies_ms,
            kv_cache_hit_rate=self.kv_cache_hit_rate,
        )
        return result


@dataclass(frozen=True)
class StepTrace:
    step: int
    start_time_ms: float
    end_time_ms: float
    latency_ms: float
    scheduled_tokens: dict[str, int]
    prefill_tokens: int
    decode_tokens: int
    kv_cache_usage: float
    prefix_cache_queries: int
    prefix_cache_hits: int


@dataclass
class SimulationResult:
    instance_id: str
    vllm_version: str
    model: str
    latency_profile: LinearLatencyProfile
    requests: list[RequestMetrics]
    steps: list[StepTrace]

    @property
    def start_time_ms(self) -> float:
        return min((request.arrival_time_ms for request in self.requests), default=0.0)

    @property
    def finish_time_ms(self) -> float:
        return max(
            (request.finish_time_ms or self.start_time_ms for request in self.requests),
            default=self.start_time_ms,
        )

    @property
    def output_tokens(self) -> int:
        return sum(request.output_tokens for request in self.requests)

    @property
    def output_throughput_tokens_per_s(self) -> float:
        duration_ms = self.finish_time_ms - self.start_time_ms
        if duration_ms <= 0:
            return 0.0
        return self.output_tokens / (duration_ms / 1000.0)

    @property
    def prefix_cache_queries(self) -> int:
        return sum(request.prompt_tokens for request in self.requests)

    @property
    def prefix_cache_hits(self) -> int:
        return sum(request.cached_tokens for request in self.requests)

    @property
    def kv_cache_hit_rate(self) -> float:
        if not self.prefix_cache_queries:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": "vllm-scheduler-simulation",
            "instance_id": self.instance_id,
            "vllm_version": self.vllm_version,
            "model": self.model,
            "latency_profile": asdict(self.latency_profile),
            "summary": {
                "request_count": len(self.requests),
                "output_tokens": self.output_tokens,
                "start_time_ms": self.start_time_ms,
                "finish_time_ms": self.finish_time_ms,
                "output_throughput_tokens_per_s": (
                    self.output_throughput_tokens_per_s
                ),
                "prefix_cache_queries": self.prefix_cache_queries,
                "prefix_cache_hits": self.prefix_cache_hits,
                "kv_cache_hit_rate": self.kv_cache_hit_rate,
            },
            "requests": [request.to_dict() for request in self.requests],
            "steps": [asdict(step) for step in self.steps],
        }


class VllmSchedulerSimulator:
    """One vLLM scheduler and KV-cache namespace for one HiSim instance."""

    def __init__(
        self,
        *,
        model: str,
        instance_id: str,
        latency_profile: LinearLatencyProfile | None = None,
        block_size: int = 16,
        num_gpu_blocks: int = 4096,
        max_model_len: int = 4096,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 256,
        generated_token_id: int = 1,
        strict_version: bool = True,
    ) -> None:
        if not instance_id:
            raise ValueError("instance_id must not be empty")
        if block_size <= 0 or num_gpu_blocks <= 0:
            raise ValueError("block_size and num_gpu_blocks must be positive")

        self.model = model
        self.instance_id = instance_id
        self.latency_profile = latency_profile or LinearLatencyProfile()
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.generated_token_id = generated_token_id
        self.vllm_version = importlib.metadata.version("vllm")
        if strict_version and self.vllm_version not in COMPATIBLE_VLLM_VERSIONS:
            raise RuntimeError(
                f"HiSim vLLM adapter supports {COMPATIBLE_VLLM_VERSIONS}, "
                f"found {self.vllm_version}"
            )

        vllm = self._load_vllm_symbols()
        self._vllm = vllm
        self._scheduler, self._block_hasher = self._build_scheduler(
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )

    @staticmethod
    def _load_vllm_symbols() -> dict[str, Any]:
        # EngineArgs probes the active platform while constructing VllmConfig.
        # The simulator has no model worker, therefore the CPU platform is the
        # correct configuration-only platform even when the target is RTX 4090.
        from vllm import platforms
        from vllm.platforms.cpu import CpuPlatform

        platforms.current_platform = CpuPlatform()

        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (
            get_request_block_hasher,
            init_none_hash,
        )
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
        )
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request
        from vllm.v1.structured_output import StructuredOutputManager

        return locals()

    def _build_scheduler(
        self,
        *,
        max_model_len: int,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> tuple[Any, Any]:
        v = self._vllm
        engine_args = v["EngineArgs"](
            model=self.model,
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=max_model_len,
            block_size=self.block_size,
            enable_prefix_caching=True,
            num_gpu_blocks_override=self.num_gpu_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=True,
            disable_log_stats=False,
        )
        config = engine_args.create_engine_config()
        # No worker profiles memory in scheduler-only mode.  Keep this field in
        # sync with the manually constructed KVCacheConfig for introspection.
        config.cache_config.num_gpu_blocks = self.num_gpu_blocks

        model_config = config.model_config
        parallel_config = config.parallel_config
        num_layers = model_config.get_num_layers(parallel_config)
        layer_names = [f"model.layers.{index}.self_attn" for index in range(num_layers)]
        cache_spec = v["FullAttentionSpec"](
            block_size=self.block_size,
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,
        )
        kv_cache_config = v["KVCacheConfig"](
            num_blocks=self.num_gpu_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[v["KVCacheGroupSpec"](layer_names, cache_spec)],
        )

        caching_hash_fn = v["get_hash_fn_by_name"](
            config.cache_config.prefix_caching_hash_algo
        )
        v["init_none_hash"](caching_hash_fn)
        block_hasher = v["get_request_block_hasher"](
            self.block_size, caching_hash_fn
        )
        scheduler = v["Scheduler"](
            config,
            kv_cache_config,
            v["StructuredOutputManager"](config),
            block_size=self.block_size,
            hash_block_size=self.block_size,
            include_finished_set=False,
            log_stats=True,
        )
        return scheduler, block_hasher

    def run(
        self, requests: Iterable[VllmRequestSpec], *, max_steps: int = 1_000_000
    ) -> SimulationResult:
        specs = sorted(
            list(requests), key=lambda request: (request.arrival_time_ms, request.request_id)
        )
        self._validate_requests(specs)
        if not specs:
            return SimulationResult(
                instance_id=self.instance_id,
                vllm_version=self.vllm_version,
                model=self.model,
                latency_profile=self.latency_profile,
                requests=[],
                steps=[],
            )

        metrics = {
            spec.request_id: RequestMetrics(
                request_id=spec.request_id,
                instance_id=spec.instance_id,
                arrival_time_ms=spec.arrival_time_ms,
                prompt_tokens=len(spec.prompt_token_ids),
            )
            for spec in specs
        }
        pending = list(specs)
        steps: list[StepTrace] = []
        clock_ms = pending[0].arrival_time_ms

        while pending or self._scheduler.has_unfinished_requests():
            while pending and pending[0].arrival_time_ms <= clock_ms:
                spec = pending.pop(0)
                self._scheduler.add_request(self._make_request(spec))

            if not self._scheduler.has_unfinished_requests():
                clock_ms = pending[0].arrival_time_ms
                continue
            if len(steps) >= max_steps:
                raise RuntimeError(f"simulation exceeded max_steps={max_steps}")

            scheduler_output = self._scheduler.schedule()
            scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)
            if not scheduled_tokens:
                raise RuntimeError("vLLM scheduler made no progress with active requests")

            prefill_tokens = 0
            decode_tokens = 0
            decode_context_tokens = 0
            for request_id, count in scheduled_tokens.items():
                request_metrics = metrics[request_id]
                if request_metrics.first_scheduled_time_ms is None:
                    request_metrics.first_scheduled_time_ms = clock_ms
                if request_metrics.output_tokens == 0:
                    prefill_tokens += count
                else:
                    decode_tokens += count
                    decode_context_tokens += self._scheduler.requests[
                        request_id
                    ].num_computed_tokens

            latency_ms = self.latency_profile.predict_ms(
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                decode_context_tokens=decode_context_tokens,
            )
            start_time_ms = clock_ms
            clock_ms += latency_ms

            ordered_ids = list(scheduled_tokens)
            sampled_token_ids = []
            for request_id in ordered_ids:
                request = self._scheduler.requests[request_id]
                sampled_token_ids.append(
                    [] if request.is_prefill_chunk else [self.generated_token_id]
                )
            model_output = self._vllm["ModelRunnerOutput"](
                req_ids=ordered_ids,
                req_id_to_index={
                    request_id: index for index, request_id in enumerate(ordered_ids)
                },
                sampled_token_ids=sampled_token_ids,
            )
            engine_outputs = self._scheduler.update_from_output(
                scheduler_output, model_output
            )

            cache_queries = 0
            cache_hits = 0
            kv_cache_usage = self._scheduler.kv_cache_manager.usage
            for client_outputs in engine_outputs.values():
                scheduler_stats = client_outputs.scheduler_stats
                if scheduler_stats is not None:
                    cache_queries += scheduler_stats.prefix_cache_stats.queries
                    cache_hits += scheduler_stats.prefix_cache_stats.hits
                    kv_cache_usage = scheduler_stats.kv_cache_usage
                for output in client_outputs.outputs:
                    request_metrics = metrics[output.request_id]
                    for _ in output.new_token_ids:
                        request_metrics.token_timestamps_ms.append(clock_ms)
                        request_metrics.output_tokens += 1
                        if request_metrics.first_token_time_ms is None:
                            request_metrics.first_token_time_ms = clock_ms
                    if output.prefill_stats is not None:
                        prefill_stats = output.prefill_stats
                        request_metrics.cached_tokens = prefill_stats.num_cached_tokens
                        request_metrics.local_cached_tokens = (
                            prefill_stats.num_local_cached_tokens
                        )
                        request_metrics.external_cached_tokens = (
                            prefill_stats.num_external_cached_tokens
                        )
                        request_metrics.computed_prompt_tokens = (
                            prefill_stats.num_computed_tokens
                        )
                    if output.finish_reason is not None:
                        request_metrics.finish_time_ms = clock_ms
                        request_metrics.finish_reason = str(output.finish_reason)

            steps.append(
                StepTrace(
                    step=len(steps),
                    start_time_ms=start_time_ms,
                    end_time_ms=clock_ms,
                    latency_ms=latency_ms,
                    scheduled_tokens=scheduled_tokens,
                    prefill_tokens=prefill_tokens,
                    decode_tokens=decode_tokens,
                    kv_cache_usage=kv_cache_usage,
                    prefix_cache_queries=cache_queries,
                    prefix_cache_hits=cache_hits,
                )
            )

        unfinished = [
            request.request_id for request in metrics.values() if request.finish_time_ms is None
        ]
        if unfinished:
            raise RuntimeError(f"requests completed without finish output: {unfinished}")
        return SimulationResult(
            instance_id=self.instance_id,
            vllm_version=self.vllm_version,
            model=self.model,
            latency_profile=self.latency_profile,
            requests=[metrics[spec.request_id] for spec in specs],
            steps=steps,
        )

    def reset_prefix_cache(self) -> None:
        """Clear this instance's vLLM block pool after all requests finish."""
        if self._scheduler.has_unfinished_requests():
            raise RuntimeError("cannot reset prefix cache with unfinished requests")
        if not self._scheduler.reset_prefix_cache():
            raise RuntimeError("vLLM refused to reset the prefix cache")

    def _make_request(self, spec: VllmRequestSpec) -> Any:
        return self._vllm["Request"](
            request_id=spec.request_id,
            prompt_token_ids=spec.prompt_token_ids,
            sampling_params=self._vllm["SamplingParams"](
                max_tokens=spec.max_tokens,
                ignore_eos=True,
                temperature=0.0,
            ),
            pooling_params=None,
            arrival_time=spec.arrival_time_ms / 1000.0,
            block_hasher=self._block_hasher,
        )

    def _validate_requests(self, specs: list[VllmRequestSpec]) -> None:
        request_ids: set[str] = set()
        for spec in specs:
            if spec.instance_id != self.instance_id:
                raise ValueError(
                    "KVCache instance isolation violation: simulator instance "
                    f"{self.instance_id!r} received request {spec.request_id!r} for "
                    f"instance {spec.instance_id!r}"
                )
            if spec.request_id in request_ids:
                raise ValueError(f"duplicate request_id: {spec.request_id}")
            request_ids.add(spec.request_id)
