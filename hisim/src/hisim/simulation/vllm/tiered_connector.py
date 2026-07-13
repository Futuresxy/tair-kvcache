"""vLLM 0.23 KVConnector for HiSim's HBM/DRAM/SSD cache model."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from .tiered_cache import (
    TieredKVCache,
    TieredKVCacheConfig,
    TieredLookupDecision,
    TieredStorePlan,
)

if TYPE_CHECKING:
    import torch

    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class HiSimTieredKVConnectorMetadata(KVConnectorMetadata):
    """Scheduler decision and transfer cost consumed by HiSimWorker."""

    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    load_latency_ms: float = 0.0
    store_latency_ms: float = 0.0
    store_plan: dict[str, Any] = field(default_factory=dict)
    cache_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_path_latency_ms(self) -> float:
        return self.load_latency_ms + self.store_latency_ms


class HiSimTieredKVConnector(KVConnectorBase_V1):
    """Synchronous simulated connector below vLLM's native HBM cache.

    The scheduler-side connector owns all simulated cache state.  Its metadata
    crosses the normal SchedulerOutput boundary and HiSimWorker accounts the
    selected read/write cost instead of copying real KV tensors.
    """

    def __init__(
        self,
        vllm_config,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        extra = dict(self._kv_transfer_config.kv_connector_extra_config)
        self.instance_id = str(extra.get("instance_id") or "")
        environment_instance_id = os.environ.get("HISIM_VLLM_INSTANCE_ID", "")
        if not self.instance_id or self.instance_id != environment_instance_id:
            raise RuntimeError(
                "tiered KV connector instance_id must equal "
                "HISIM_VLLM_INSTANCE_ID"
            )
        self.block_size = vllm_config.cache_config.block_size
        self.trace_path = os.environ.get("HISIM_VLLM_TRACE_PATH")
        self._requests: dict[str, Request] = {}
        self._stored_blocks_by_request: dict[str, int] = {}
        self._pending_decisions: dict[str, TieredLookupDecision] = {}
        self.cache: TieredKVCache | None = None

        if role == KVConnectorRole.SCHEDULER:
            config = TieredKVCacheConfig.from_mapping(extra.get("tiered_config"))
            kv_bytes_per_block = sum(
                len(group.layer_names) * group.kv_cache_spec.page_size_bytes
                for group in kv_cache_config.kv_cache_groups
            )
            model_config = vllm_config.model_config
            namespace = json.dumps(
                {
                    "instance_id": self.instance_id,
                    "model": model_config.served_model_name,
                    "dtype": str(model_config.dtype),
                    "block_size": self.block_size,
                    "tp": vllm_config.parallel_config.tensor_parallel_size,
                    "pp": vllm_config.parallel_config.pipeline_parallel_size,
                },
                sort_keys=True,
                default=str,
            )
            self.cache = TieredKVCache(
                instance_id=self.instance_id,
                namespace=namespace,
                block_size=self.block_size,
                kv_bytes_per_block=kv_bytes_per_block,
                config=config,
            )
            self._write_trace(
                {
                    "event": "tiered_cache_initialized",
                    "cache_snapshot": self.cache.snapshot(),
                }
            )

    # Worker-side tensor transfer primitives are intentionally no-ops.  The
    # HiSim Worker consumes connector metadata and simulates their critical path.
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def on_new_request(self, request: "Request") -> None:
        if self.role == KVConnectorRole.SCHEDULER:
            self._requests[request.request_id] = request

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.role != KVConnectorRole.SCHEDULER or self.cache is None:
            return 0, False
        self._requests[request.request_id] = request
        decision = self.cache.plan_lookup(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids or []),
            local_hbm_tokens=num_computed_tokens,
        )
        self._pending_decisions[request.request_id] = decision
        return decision.selected_external_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.role == KVConnectorRole.SCHEDULER:
            self._requests[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> HiSimTieredKVConnectorMetadata:
        if self.role != KVConnectorRole.SCHEDULER or self.cache is None:
            return HiSimTieredKVConnectorMetadata()

        store_plan = TieredStorePlan()
        for request_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self._requests.get(request_id)
            if request is None:
                continue
            token_ids = list(request.all_token_ids)
            computed_after_step = min(
                len(token_ids), request.num_computed_tokens + scheduled_tokens
            )
            block_keys = self.cache.prefix_block_keys(
                token_ids, upto_tokens=computed_after_step
            )
            start = min(
                self._stored_blocks_by_request.get(request_id, 0), len(block_keys)
            )
            request_store_plan = self.cache.store_prefix_blocks(block_keys[start:])
            store_plan.merge(request_store_plan)
            self._stored_blocks_by_request[request_id] = len(block_keys)

        decisions = {
            request_id: decision.trace_dict()
            for request_id, decision in self._pending_decisions.items()
        }
        load_latency_ms = sum(
            decision.effective_load_latency_ms
            for decision in self._pending_decisions.values()
        )
        metadata = HiSimTieredKVConnectorMetadata(
            decisions=decisions,
            load_latency_ms=load_latency_ms,
            store_latency_ms=store_plan.effective_store_latency_ms,
            store_plan=asdict(store_plan),
            cache_snapshot=self.cache.snapshot(),
        )
        self._pending_decisions.clear()
        self.cache.write_metrics()
        self._write_trace(
            {
                "event": "tiered_cache_step",
                "request_ids": list(scheduler_output.num_scheduled_tokens),
                "decisions": decisions,
                "load_latency_ms": metadata.load_latency_ms,
                "store_latency_ms": metadata.store_latency_ms,
                "store_plan": metadata.store_plan,
                "cache_snapshot": metadata.cache_snapshot,
            }
        )
        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._requests.pop(request.request_id, None)
        self._stored_blocks_by_request.pop(request.request_id, None)
        self._pending_decisions.pop(request.request_id, None)
        return False, None

    def reset_cache(self) -> bool:
        if self.cache is not None:
            self.cache.reset()
            self._write_trace(
                {
                    "event": "tiered_cache_reset",
                    "cache_snapshot": self.cache.snapshot(),
                }
            )
        self._stored_blocks_by_request.clear()
        self._pending_decisions.clear()
        return True

    def shutdown(self) -> None:
        if self.cache is not None:
            self.cache.write_metrics()

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if not self.trace_path:
            return
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "backend": "vllm-hisim-tiered-kv-connector",
            "instance_id": self.instance_id,
            "monotonic_time": time.monotonic(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
