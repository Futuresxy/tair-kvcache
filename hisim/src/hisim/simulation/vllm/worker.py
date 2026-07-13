"""vLLM 0.23 worker that replaces model execution with HiSim prediction.

The worker is loaded through vLLM's public ``worker_cls`` configuration.  The
OpenAI frontend, EngineCore, Scheduler and KVCacheManager remain unchanged;
only device/model/KV-tensor allocation and kernel execution are replaced.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from .scheduler_simulator import LinearLatencyProfile
from .simulate import load_latency_profile


class HiSimWorker(WorkerBase):
    """A no-kernel vLLM worker driven by a HiSim latency profile."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.instance_id = os.environ.get("HISIM_VLLM_INSTANCE_ID", "")
        if not self.instance_id:
            raise RuntimeError(
                "HISIM_VLLM_INSTANCE_ID is required to preserve KVCache isolation"
            )
        profile_path = os.environ.get("HISIM_VLLM_LATENCY_PROFILE")
        self.latency_profile = (
            load_latency_profile(Path(profile_path))
            if profile_path
            else LinearLatencyProfile()
        )
        self.execution_mode = os.environ.get(
            "HISIM_VLLM_EXECUTION_MODE", "wall_clock"
        ).lower()
        if self.execution_mode not in {"wall_clock", "fast_forward"}:
            raise ValueError(
                "HISIM_VLLM_EXECUTION_MODE must be wall_clock or fast_forward"
            )
        self.time_scale = float(os.environ.get("HISIM_VLLM_TIME_SCALE", "1"))
        if self.time_scale <= 0:
            raise ValueError("HISIM_VLLM_TIME_SCALE must be positive")
        self.trace_path = os.environ.get("HISIM_VLLM_TRACE_PATH")
        self.num_simulated_blocks = int(
            os.environ.get(
                "HISIM_VLLM_NUM_GPU_BLOCKS",
                str(self.cache_config.num_gpu_blocks_override or 4096),
            )
        )
        if self.num_simulated_blocks <= 0:
            raise ValueError("HISIM_VLLM_NUM_GPU_BLOCKS must be positive")
        vocab_size = self.model_config.get_vocab_size()
        requested_token_id = int(os.environ.get("HISIM_VLLM_TOKEN_ID", "9707"))
        self.generated_token_id = min(max(0, requested_token_id), vocab_size - 1)
        self._request_prompt_lengths: dict[str, int] = {}
        self._step = 0
        self._model = nn.Identity()
        self._kv_cache_config: KVCacheConfig | None = None

    def init_device(self) -> None:
        # Configuration-only CPU state: no CUDA context, NCCL or allocator.
        self.device = torch.device("cpu")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # The frontend still loads the tokenizer; the worker loads no weights.
        return

    def get_model(self) -> nn.Module:
        return self._model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        spec = FullAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=self.model_config.get_num_kv_heads(self.parallel_config),
            head_size=self.model_config.get_head_size(),
            dtype=self.model_config.dtype,
        )
        return {
            f"model.layers.{layer}.self_attn": spec for layer in range(num_layers)
        }

    def determine_available_memory(self) -> int:
        # vLLM derives its scheduler block pool from this byte count.  No tensor
        # is allocated later, but the native cache planner still runs unchanged.
        bytes_per_block = sum(
            spec.page_size_bytes for spec in self.get_kv_cache_spec().values()
        )
        return bytes_per_block * self.num_simulated_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self._kv_cache_config = kv_cache_config
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self._write_trace(
            {
                "event": "initialized",
                "num_blocks": kv_cache_config.num_blocks,
                "profile": asdict(self.latency_profile),
            }
        )

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        for request_id in scheduler_output.finished_req_ids:
            self._request_prompt_lengths.pop(request_id, None)

        before_computed: dict[str, int] = {}
        output_counts: dict[str, int] = {}
        for request in scheduler_output.scheduled_new_reqs:
            prompt_length = len(request.prompt_token_ids or [])
            self._request_prompt_lengths[request.req_id] = prompt_length
            before_computed[request.req_id] = request.num_computed_tokens
            output_counts[request.req_id] = 0

        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            before_computed[request_id] = cached.num_computed_tokens[index]
            output_counts[request_id] = cached.num_output_tokens[index]

        prefill_tokens = 0
        decode_tokens = 0
        decode_context_tokens = 0
        sampled_token_ids: list[list[int]] = []
        request_ids = list(scheduler_output.num_scheduled_tokens)
        for request_id in request_ids:
            scheduled = scheduler_output.num_scheduled_tokens[request_id]
            num_output_tokens = output_counts.get(request_id, 0)
            if num_output_tokens == 0:
                prefill_tokens += scheduled
                prompt_length = self._request_prompt_lengths[request_id]
                prefill_complete = (
                    before_computed.get(request_id, 0) + scheduled >= prompt_length
                )
                sampled_token_ids.append(
                    [self.generated_token_id] if prefill_complete else []
                )
            else:
                decode_tokens += scheduled
                decode_context_tokens += before_computed.get(request_id, 0) + scheduled
                sampled_token_ids.append([self.generated_token_id])

        predicted_latency_ms = self.latency_profile.predict_ms(
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            decode_context_tokens=decode_context_tokens,
        )
        wall_start = time.perf_counter()
        if self.execution_mode == "wall_clock":
            time.sleep(predicted_latency_ms / self.time_scale / 1000.0)
        wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
        self._write_trace(
            {
                "event": "execute_model",
                "step": self._step,
                "request_ids": request_ids,
                "scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "decode_context_tokens": decode_context_tokens,
                "predicted_latency_ms": predicted_latency_ms,
                "wall_elapsed_ms": wall_elapsed_ms,
                "execution_mode": self.execution_mode,
            }
        )
        self._step += 1
        return ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(request_ids)
            },
            sampled_token_ids=sampled_token_ids,
        )

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        raise RuntimeError("HiSimWorker returns sampled tokens from execute_model")

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    def get_cache_block_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len

    def take_draft_token_ids(self):
        return None

    def add_lora(self, lora_request) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if not self.trace_path:
            return
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "backend": "vllm-engine-hisim-worker",
            "instance_id": self.instance_id,
            "monotonic_time": time.monotonic(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
