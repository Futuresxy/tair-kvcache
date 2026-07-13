"""SGLang HiCache adapter for HiSim's shared tiered-cache policy core."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from hisim.simulation.tiered_cache import (
    TieredKVCache,
    TieredKVCacheConfig,
    TieredLookupDecision,
)


@dataclass(frozen=True)
class SGLangTieredCacheSettings:
    """Framework binding around the shared policy configuration."""

    instance_id: str
    config: TieredKVCacheConfig
    decision_trace_path: str | None = None
    reset_on_start: bool = False


@dataclass(frozen=True)
class _RequestContext:
    local_hbm_tokens: int
    local_dram_tokens: int
    prompt_tokens: int


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tiered KV cache configuration must be a JSON object")
    return payload


def load_sglang_tiered_cache_settings(
    config_path: str | Path | None = None,
) -> SGLangTieredCacheSettings | None:
    """Load an explicitly enabled SGLang tiered-cache configuration.

    A standalone file from ``HISIM_SGLANG_TIERED_KV_CONFIG`` may contain either
    the policy mapping directly or a top-level ``tiered_kv_cache`` object.  If
    it is absent, the same object is read from ``HISIM_CONFIG_PATH``.
    """

    standalone_path = os.getenv("HISIM_SGLANG_TIERED_KV_CONFIG")
    source_path = standalone_path or config_path or os.getenv("HISIM_CONFIG_PATH")
    if not source_path or not Path(source_path).exists():
        return None
    payload = _load_json(source_path)
    raw = payload.get("tiered_kv_cache")
    if raw is None and standalone_path:
        raw = payload
    if not isinstance(raw, Mapping) or not raw.get("enabled", True):
        return None

    values = dict(raw)
    values.pop("enabled", None)
    configured_instance_id = str(values.pop("instance_id", "") or "")
    environment_instance_id = os.getenv("HISIM_SGLANG_INSTANCE_ID", "")
    if (
        configured_instance_id
        and environment_instance_id
        and configured_instance_id != environment_instance_id
    ):
        raise RuntimeError(
            "tiered KV cache instance_id must equal HISIM_SGLANG_INSTANCE_ID"
        )
    instance_id = configured_instance_id or environment_instance_id
    if not instance_id:
        raise RuntimeError(
            "HISIM_SGLANG_INSTANCE_ID or tiered_kv_cache.instance_id is required "
            "to preserve KVCache Instance isolation"
        )

    decision_trace_path = values.pop("decision_trace_path", None)
    reset_on_start = bool(values.pop("reset_on_start", False))
    dram_capacity_gb = values.pop("dram_capacity_gb", None)
    ssd_capacity_gb = values.pop("ssd_capacity_gb", None)
    if dram_capacity_gb is not None and "dram_capacity_bytes" not in values:
        values["dram_capacity_bytes"] = int(float(dram_capacity_gb) * 1024**3)
    if ssd_capacity_gb is not None and "ssd_capacity_bytes" not in values:
        values["ssd_capacity_bytes"] = int(float(ssd_capacity_gb) * 1024**3)

    platform = payload.get("platform", {}) if not standalone_path else {}
    if isinstance(platform, Mapping):
        fallbacks = {
            "dram_read_bandwidth_gbps": platform.get("memory_read_bandwidth_gb"),
            "dram_write_bandwidth_gbps": platform.get("memory_write_bandwidth_gb"),
            "ssd_read_bandwidth_gbps": platform.get("disk_read_bandwidth_gb"),
            "ssd_write_bandwidth_gbps": platform.get("disk_write_bandwidth_gb"),
        }
        for name, value in fallbacks.items():
            if name not in values and value is not None:
                values[name] = float(value)

    output_dir = Path(os.getenv("HISIM_OUTPUT_DIR", "/tmp/hisim/simulation"))
    if "metrics_path" not in values:
        values["metrics_path"] = str(output_dir / "tiered_cache_metrics.json")
    if decision_trace_path is None:
        decision_trace_path = str(output_dir / "tiered_cache_decisions.jsonl")

    config = TieredKVCacheConfig.from_mapping(values)
    if (
        config.dram_capacity_bytes == 0
        and config.ssd_capacity_bytes == 0
        and not config.dram_capacity_blocks
        and not config.ssd_capacity_blocks
    ):
        raise ValueError(
            "an enabled SGLang tiered cache requires non-zero DRAM or SSD capacity"
        )
    return SGLangTieredCacheSettings(
        instance_id=instance_id,
        config=config,
        decision_trace_path=decision_trace_path,
        reset_on_start=reset_on_start,
    )


class SGLangTieredCacheBackend:
    """Storage-backend surface consumed by SGLang's HiCacheController.

    SGLang retains its HBM radix tree; its native host pool is transport
    staging rather than logical unbounded DRAM.  This adapter translates page
    hashes and request context into the shared policy model, then returns only
    the prefix selected for retrieval.
    """

    def __init__(
        self,
        *,
        settings: SGLangTieredCacheSettings,
        storage_config: Any,
        mem_pool_host: Any,
    ) -> None:
        self.settings = settings
        self.instance_id = settings.instance_id
        self.page_size = int(mem_pool_host.page_size)
        device_pool = getattr(mem_pool_host, "device_pool", None)
        kv_bytes_per_token = getattr(
            device_pool,
            "hisim_kv_bytes_per_token",
            mem_pool_host.size_per_token,
        )
        self.kv_bytes_per_block = int(kv_bytes_per_token) * self.page_size
        if self.page_size <= 0 or self.kv_bytes_per_block <= 0:
            raise ValueError("SGLang page and KV block sizes must be positive")

        namespace = json.dumps(
            {
                "instance_id": self.instance_id,
                "framework": "sglang",
                "model": getattr(storage_config, "model_name", None),
                "dtype": str(getattr(mem_pool_host, "dtype", "unknown")),
                "page_size": self.page_size,
                "kv_bytes_per_block": self.kv_bytes_per_block,
                "tp_rank": getattr(storage_config, "tp_rank", 0),
                "tp_size": getattr(storage_config, "tp_size", 1),
            },
            sort_keys=True,
        )
        state_path = settings.config.ssd_state_path
        if settings.reset_on_start and state_path:
            Path(state_path).unlink(missing_ok=True)
        self.cache = TieredKVCache(
            instance_id=self.instance_id,
            namespace=namespace,
            block_size=self.page_size,
            kv_bytes_per_block=self.kv_bytes_per_block,
            config=settings.config,
        )
        self.mem_pool_host = mem_pool_host
        self._request_context: dict[str, _RequestContext] = {}
        self._last_decisions: dict[str, TieredLookupDecision] = {}
        self._completed_prefetch_tokens: dict[str, int] = {}
        self._prefetch_pages: list[int] = []
        self._backup_pages: list[int] = []

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        if mem_pool_host is not self.mem_pool_host:
            raise RuntimeError("SGLang tiered backend host pool changed unexpectedly")

    def register_request_context(
        self,
        request_id: str,
        *,
        local_hbm_tokens: int,
        local_dram_tokens: int,
        prompt_tokens: int,
    ) -> None:
        self._request_context[request_id] = _RequestContext(
            local_hbm_tokens=max(0, int(local_hbm_tokens)),
            local_dram_tokens=max(0, int(local_dram_tokens)),
            prompt_tokens=max(1, int(prompt_tokens)),
        )

    def touch_resident_blocks(self, block_keys: Iterable[str]) -> None:
        self.cache.touch_blocks(block_keys)

    def lookup_tier(self, block_key: str) -> str | None:
        """Expose logical placement to the SGLang host-radix adapter."""

        return self.cache.lookup_tier(block_key)

    def plan_prefetch(
        self, request_id: str, block_keys: Iterable[str]
    ) -> TieredLookupDecision:
        keys = list(block_keys)
        context = self._request_context.pop(request_id, None)
        if context is None:
            context = _RequestContext(
                local_hbm_tokens=0,
                local_dram_tokens=0,
                prompt_tokens=len(keys) * self.page_size + 1,
            )
        decision = self.cache.plan_lookup_by_block_keys(
            request_id=request_id,
            block_keys=keys,
            local_hbm_tokens=context.local_hbm_tokens,
            local_dram_tokens=context.local_dram_tokens,
            prompt_tokens=context.prompt_tokens,
            promote_selected=False,
        )
        self._last_decisions[request_id] = decision
        self._prefetch_pages.append(decision.selected_external_blocks)
        self._write_trace(
            {
                "event": "lookup",
                "decision": decision.trace_dict(),
                "cache_snapshot": self.cache.snapshot(),
            }
        )
        return decision

    def record_prefetch_completion(
        self, request_id: str, completed_tokens: int
    ) -> None:
        """Commit only pages completed by SGLang's best-effort prefetch."""

        decision = self._last_decisions.get(request_id)
        if decision is None:
            return
        completed_tokens = min(
            max(0, int(completed_tokens)),
            decision.selected_external_tokens,
        )
        previous_tokens = self._completed_prefetch_tokens.get(request_id, 0)
        if completed_tokens <= previous_tokens:
            return
        previous_blocks = previous_tokens // self.page_size
        completed_blocks = completed_tokens // self.page_size
        self.cache.commit_selected_prefix(
            decision,
            start_block=previous_blocks,
            end_block=completed_blocks,
        )
        self._completed_prefetch_tokens[request_id] = completed_tokens

    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        return self.batch_set([key], [value])

    def batch_set(
        self,
        keys: list[str],
        values: Any = None,
        extra_info: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        plan = self.cache.store_prefix_blocks(keys)
        self._backup_pages.append(plan.stored_blocks)
        if plan.stored_blocks:
            self._write_trace(
                {
                    "event": "store",
                    "stored_blocks": plan.stored_blocks,
                    "dram_write_blocks": plan.dram_write_blocks,
                    "ssd_write_blocks": plan.ssd_write_blocks,
                    "cache_snapshot": self.cache.snapshot(),
                }
            )
        return True

    def exists(self, key: str) -> bool:
        return self.cache.lookup_tier(key) is not None

    def batch_exists(self, keys: list[str], extra_info: Any = None) -> int:
        return self.cache.contiguous_available_blocks(keys)

    def clear(self) -> bool:
        self._request_context.clear()
        self._last_decisions.clear()
        self._completed_prefetch_tokens.clear()
        self._prefetch_pages.clear()
        self._backup_pages.clear()
        self.cache.reset()
        return True

    def reset(self) -> None:
        self.clear()

    def close(self) -> None:
        self.cache.write_metrics()

    def get_stats(self):
        # SGLang imports this concrete dataclass in its Prometheus collector.
        from sglang.srt.metrics.collector import StorageMetrics

        stats = StorageMetrics(
            prefetch_pgs=self._prefetch_pages.copy(),
            backup_pgs=self._backup_pages.copy(),
        )
        self._prefetch_pages.clear()
        self._backup_pages.clear()
        return stats

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.cache.snapshot()
        snapshot["framework"] = "sglang"
        snapshot["last_decisions"] = {
            request_id: decision.trace_dict()
            for request_id, decision in self._last_decisions.items()
        }
        planned_tokens = sum(
            decision.selected_external_tokens
            for decision in self._last_decisions.values()
        )
        completed_tokens = sum(self._completed_prefetch_tokens.values())
        completed_dram_blocks = 0
        completed_ssd_blocks = 0
        for request_id, completed in self._completed_prefetch_tokens.items():
            decision = self._last_decisions.get(request_id)
            if decision is None:
                continue
            count = completed // self.page_size
            tiers = decision.selected_tiers[:count]
            completed_dram_blocks += tiers.count("dram")
            completed_ssd_blocks += tiers.count("ssd")
        cacheable_tokens = sum(
            decision.cacheable_tokens for decision in self._last_decisions.values()
        )
        hbm_hit_tokens = sum(
            decision.local_hbm_tokens for decision in self._last_decisions.values()
        )
        actual_hit_tokens = hbm_hit_tokens + completed_tokens
        snapshot["runtime_metrics"] = {
            "planned_prefetch_tokens": planned_tokens,
            "completed_prefetch_tokens": completed_tokens,
            "prefetch_completion_rate": (
                completed_tokens / planned_tokens if planned_tokens else 0.0
            ),
            "hbm_hit_tokens": hbm_hit_tokens,
            "dram_hit_tokens": completed_dram_blocks * self.page_size,
            "ssd_hit_tokens": completed_ssd_blocks * self.page_size,
            "actual_hit_tokens": actual_hit_tokens,
            "actual_hit_rate": (
                actual_hit_tokens / cacheable_tokens if cacheable_tokens else 0.0
            ),
        }
        return snapshot

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if not self.settings.decision_trace_path:
            return
        path = Path(self.settings.decision_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "backend": "sglang-hisim-tiered-kv-adapter",
            "instance_id": self.instance_id,
            "monotonic_time": time.monotonic(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_sglang_tiered_cache_backend(
    *, storage_config: Any, mem_pool_host: Any
) -> SGLangTieredCacheBackend | None:
    settings = load_sglang_tiered_cache_settings()
    if settings is None:
        return None
    return SGLangTieredCacheBackend(
        settings=settings,
        storage_config=storage_config,
        mem_pool_host=mem_pool_host,
    )


__all__ = [
    "SGLangTieredCacheBackend",
    "SGLangTieredCacheSettings",
    "create_sglang_tiered_cache_backend",
    "load_sglang_tiered_cache_settings",
]
