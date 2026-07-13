"""Deterministic HBM/DRAM/SSD KV-cache policy model for HiSim.

HBM ownership remains in vLLM's native KVCacheManager.  This module models
the external DRAM and SSD tiers, including eviction, promotion, persistence,
transfer cost, and the retrieve-versus-recompute decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


_EVICTION_POLICIES = {"lru", "fifo"}
_PREFETCH_POLICIES = {"none", "always", "cost_aware"}
_WRITE_POLICIES = {"none", "write_through"}


@dataclass
class TieredKVCacheConfig:
    """Configuration for the simulated external KV-cache tiers."""

    dram_capacity_bytes: int = 0
    ssd_capacity_bytes: int = 0
    dram_capacity_blocks: int | None = None
    ssd_capacity_blocks: int | None = None
    eviction_policy: str = "lru"
    prefetch_policy: str = "cost_aware"
    write_policy: str = "write_through"
    recompute_ms_per_token: float = 0.05
    recompute_ms_per_token_squared: float = 0.0
    min_savings_ms: float = 0.0
    max_prefetch_blocks: int = 0
    dram_read_bandwidth_gbps: float = 50.0
    dram_write_bandwidth_gbps: float = 40.0
    ssd_read_bandwidth_gbps: float = 7.0
    ssd_write_bandwidth_gbps: float = 5.0
    dram_read_latency_ms: float = 0.01
    dram_write_latency_ms: float = 0.01
    ssd_read_latency_ms: float = 0.08
    ssd_write_latency_ms: float = 0.08
    prefetch_overlap_fraction: float = 0.0
    write_overlap_fraction: float = 1.0
    parallel_tier_reads: bool = False
    ssd_state_path: str | None = None
    metrics_path: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TieredKVCacheConfig":
        config = cls(**dict(values or {}))
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TieredKVCacheConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        self.eviction_policy = self.eviction_policy.lower()
        self.prefetch_policy = self.prefetch_policy.lower()
        self.write_policy = self.write_policy.lower()
        if self.eviction_policy not in _EVICTION_POLICIES:
            raise ValueError("eviction_policy must be lru or fifo")
        if self.prefetch_policy not in _PREFETCH_POLICIES:
            raise ValueError("prefetch_policy must be none, always, or cost_aware")
        if self.write_policy not in _WRITE_POLICIES:
            raise ValueError("write_policy must be none or write_through")
        integer_values = {
            "dram_capacity_bytes": self.dram_capacity_bytes,
            "ssd_capacity_bytes": self.ssd_capacity_bytes,
            "max_prefetch_blocks": self.max_prefetch_blocks,
        }
        if self.dram_capacity_blocks is not None:
            integer_values["dram_capacity_blocks"] = self.dram_capacity_blocks
        if self.ssd_capacity_blocks is not None:
            integer_values["ssd_capacity_blocks"] = self.ssd_capacity_blocks
        for name, value in integer_values.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        nonnegative_values = {
            "recompute_ms_per_token": self.recompute_ms_per_token,
            "recompute_ms_per_token_squared": (
                self.recompute_ms_per_token_squared
            ),
            "min_savings_ms": self.min_savings_ms,
            "dram_read_latency_ms": self.dram_read_latency_ms,
            "dram_write_latency_ms": self.dram_write_latency_ms,
            "ssd_read_latency_ms": self.ssd_read_latency_ms,
            "ssd_write_latency_ms": self.ssd_write_latency_ms,
        }
        for name, value in nonnegative_values.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        for name in (
            "dram_read_bandwidth_gbps",
            "dram_write_bandwidth_gbps",
            "ssd_read_bandwidth_gbps",
            "ssd_write_bandwidth_gbps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("prefetch_overlap_fraction", "write_overlap_fraction"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class TieredBlock:
    key: str
    inserted_at: int
    last_accessed_at: int
    access_count: int = 0


class _CacheTier:
    def __init__(self, name: str, capacity_blocks: int, policy: str) -> None:
        self.name = name
        self.capacity_blocks = capacity_blocks
        self.policy = policy
        self.entries: dict[str, TieredBlock] = {}

    def peek(self, key: str) -> TieredBlock | None:
        return self.entries.get(key)

    def access(self, key: str, sequence: int) -> TieredBlock | None:
        entry = self.entries.get(key)
        if entry is not None:
            entry.last_accessed_at = sequence
            entry.access_count += 1
        return entry

    def remove(self, key: str) -> TieredBlock | None:
        return self.entries.pop(key, None)

    def put(self, entry: TieredBlock, sequence: int) -> list[TieredBlock]:
        existing = self.entries.get(entry.key)
        if existing is not None:
            existing.last_accessed_at = sequence
            return []
        if self.capacity_blocks == 0:
            return [entry]
        entry.inserted_at = sequence
        entry.last_accessed_at = sequence
        self.entries[entry.key] = entry
        evicted: list[TieredBlock] = []
        while len(self.entries) > self.capacity_blocks:
            if self.policy == "fifo":
                victim = min(self.entries.values(), key=lambda item: item.inserted_at)
            else:
                victim = min(
                    self.entries.values(), key=lambda item: item.last_accessed_at
                )
            evicted.append(self.entries.pop(victim.key))
        return evicted

    def clear(self) -> None:
        self.entries.clear()


@dataclass
class TieredCacheMetrics:
    lookup_requests: int = 0
    cache_decisions: int = 0
    recompute_decisions: int = 0
    cacheable_query_tokens: int = 0
    hbm_hit_tokens: int = 0
    dram_candidate_tokens: int = 0
    ssd_candidate_tokens: int = 0
    dram_hit_tokens: int = 0
    ssd_hit_tokens: int = 0
    recompute_tokens: int = 0
    dram_read_bytes: int = 0
    ssd_read_bytes: int = 0
    dram_write_bytes: int = 0
    ssd_write_bytes: int = 0
    dram_evicted_blocks: int = 0
    ssd_evicted_blocks: int = 0
    promoted_blocks: int = 0
    stored_blocks: int = 0
    estimated_load_latency_ms: float = 0.0
    estimated_store_latency_ms: float = 0.0
    estimated_recompute_avoided_ms: float = 0.0


@dataclass
class TieredLookupDecision:
    request_id: str
    decision: str
    local_hbm_tokens: int
    cacheable_tokens: int
    contiguous_external_blocks: int
    available_dram_blocks: int
    available_ssd_blocks: int
    selected_external_blocks: int
    selected_external_tokens: int
    dram_blocks: int
    ssd_blocks: int
    baseline_recompute_tokens: int
    selected_recompute_tokens: int
    best_candidate_blocks: int
    best_candidate_load_latency_ms: float
    best_candidate_total_latency_ms: float
    best_candidate_savings_ms: float
    raw_load_latency_ms: float
    effective_load_latency_ms: float
    baseline_recompute_latency_ms: float
    selected_recompute_latency_ms: float
    estimated_savings_ms: float
    selected_keys: list[str] = field(default_factory=list, repr=False)

    def trace_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("selected_keys", None)
        return values


@dataclass
class TieredStorePlan:
    stored_blocks: int = 0
    dram_write_blocks: int = 0
    ssd_write_blocks: int = 0
    dram_evicted_blocks: int = 0
    ssd_evicted_blocks: int = 0
    raw_store_latency_ms: float = 0.0
    effective_store_latency_ms: float = 0.0

    def merge(self, other: "TieredStorePlan") -> None:
        for name in (
            "stored_blocks",
            "dram_write_blocks",
            "ssd_write_blocks",
            "dram_evicted_blocks",
            "ssd_evicted_blocks",
            "raw_store_latency_ms",
            "effective_store_latency_ms",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))


class TieredKVCache:
    """Exclusive DRAM/SSD cache below vLLM's native HBM cache."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        instance_id: str,
        namespace: str,
        block_size: int,
        kv_bytes_per_block: int,
        config: TieredKVCacheConfig,
    ) -> None:
        if not instance_id:
            raise ValueError("instance_id must not be empty")
        if block_size <= 0 or kv_bytes_per_block <= 0:
            raise ValueError("block_size and kv_bytes_per_block must be positive")
        config.validate()
        self.instance_id = instance_id
        self.namespace = namespace
        self.block_size = block_size
        self.kv_bytes_per_block = kv_bytes_per_block
        self.config = config
        dram_blocks = self._capacity_blocks(
            config.dram_capacity_blocks, config.dram_capacity_bytes
        )
        ssd_blocks = self._capacity_blocks(
            config.ssd_capacity_blocks, config.ssd_capacity_bytes
        )
        self.dram = _CacheTier("dram", dram_blocks, config.eviction_policy)
        self.ssd = _CacheTier("ssd", ssd_blocks, config.eviction_policy)
        self.metrics = TieredCacheMetrics()
        self._sequence = 0
        self._load_ssd_state()

    @property
    def enabled(self) -> bool:
        return self.dram.capacity_blocks > 0 or self.ssd.capacity_blocks > 0

    def _capacity_blocks(self, explicit: int | None, capacity_bytes: int) -> int:
        if explicit is not None:
            return explicit
        return capacity_bytes // self.kv_bytes_per_block

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _initial_hasher(self) -> Any:
        hasher = hashlib.sha256()
        hasher.update(self.namespace.encode("utf-8"))
        hasher.update(b"\0")
        return hasher

    def prefix_block_keys(
        self, token_ids: Iterable[int], upto_tokens: int | None = None
    ) -> list[str]:
        tokens = list(token_ids)
        if upto_tokens is not None:
            tokens = tokens[:upto_tokens]
        num_full_blocks = len(tokens) // self.block_size
        hasher = self._initial_hasher()
        keys: list[str] = []
        for block_index in range(num_full_blocks):
            start = block_index * self.block_size
            block = tokens[start : start + self.block_size]
            hasher.update(struct.pack(f"<{len(block)}q", *block))
            keys.append(hasher.hexdigest())
        return keys

    def lookup_tier(self, key: str) -> str | None:
        if self.dram.peek(key) is not None:
            return "dram"
        if self.ssd.peek(key) is not None:
            return "ssd"
        return None

    def plan_lookup(
        self,
        *,
        request_id: str,
        prompt_token_ids: list[int],
        local_hbm_tokens: int,
    ) -> TieredLookupDecision:
        max_cacheable_tokens = (
            max(0, len(prompt_token_ids) - 1) // self.block_size * self.block_size
        )
        local_hbm_tokens = min(
            max_cacheable_tokens,
            max(0, local_hbm_tokens // self.block_size * self.block_size),
        )
        all_keys = self.prefix_block_keys(
            prompt_token_ids, upto_tokens=max_cacheable_tokens
        )
        first_external_block = local_hbm_tokens // self.block_size
        candidates: list[tuple[str, str]] = []
        for key in all_keys[first_external_block:]:
            tier = self.lookup_tier(key)
            if tier is None:
                break
            candidates.append((key, tier))
            if (
                self.config.max_prefetch_blocks > 0
                and len(candidates) >= self.config.max_prefetch_blocks
            ):
                break

        baseline_tokens = max(1, len(prompt_token_ids) - local_hbm_tokens)
        baseline_compute_ms = self._recompute_latency_ms(
            start_token=local_hbm_tokens,
            end_token=len(prompt_token_ids),
        )
        selected_count = 0
        selected_raw_load_ms = 0.0
        selected_effective_load_ms = 0.0
        best_candidate_count = 0
        best_candidate_raw_load_ms = 0.0
        best_candidate_effective_load_ms = 0.0
        best_candidate_total_ms = baseline_compute_ms

        for count in range(1, len(candidates) + 1):
            raw_load_ms = self._read_latency_ms(candidates[:count])
            effective_load_ms = raw_load_ms * (
                1.0 - self.config.prefetch_overlap_fraction
            )
            total_ms = (
                effective_load_ms
                + self._recompute_latency_ms(
                    start_token=local_hbm_tokens + count * self.block_size,
                    end_token=len(prompt_token_ids),
                )
            )
            if best_candidate_count == 0 or total_ms < best_candidate_total_ms:
                best_candidate_count = count
                best_candidate_total_ms = total_ms
                best_candidate_raw_load_ms = raw_load_ms
                best_candidate_effective_load_ms = effective_load_ms

        if self.enabled and self.config.prefetch_policy == "always":
            selected_count = len(candidates)
        elif self.enabled and self.config.prefetch_policy == "cost_aware":
            if (
                best_candidate_count > 0
                and baseline_compute_ms - best_candidate_total_ms
                >= self.config.min_savings_ms
            ):
                selected_count = best_candidate_count
                selected_raw_load_ms = best_candidate_raw_load_ms
                selected_effective_load_ms = best_candidate_effective_load_ms

        if selected_count and self.config.prefetch_policy == "always":
            selected_raw_load_ms = self._read_latency_ms(candidates[:selected_count])
            selected_effective_load_ms = selected_raw_load_ms * (
                1.0 - self.config.prefetch_overlap_fraction
            )
        selected = candidates[:selected_count]
        available_dram_blocks = sum(tier == "dram" for _, tier in candidates)
        available_ssd_blocks = sum(tier == "ssd" for _, tier in candidates)
        dram_blocks = sum(tier == "dram" for _, tier in selected)
        ssd_blocks = sum(tier == "ssd" for _, tier in selected)
        selected_tokens = selected_count * self.block_size
        selected_recompute_tokens = max(1, baseline_tokens - selected_tokens)
        selected_compute_ms = self._recompute_latency_ms(
            start_token=local_hbm_tokens + selected_tokens,
            end_token=len(prompt_token_ids),
        )
        savings_ms = baseline_compute_ms - (
            selected_effective_load_ms + selected_compute_ms
        )
        decision = TieredLookupDecision(
            request_id=request_id,
            decision="cache" if selected_count else "recompute",
            local_hbm_tokens=local_hbm_tokens,
            cacheable_tokens=max_cacheable_tokens,
            contiguous_external_blocks=len(candidates),
            available_dram_blocks=available_dram_blocks,
            available_ssd_blocks=available_ssd_blocks,
            selected_external_blocks=selected_count,
            selected_external_tokens=selected_tokens,
            dram_blocks=dram_blocks,
            ssd_blocks=ssd_blocks,
            baseline_recompute_tokens=baseline_tokens,
            selected_recompute_tokens=selected_recompute_tokens,
            best_candidate_blocks=best_candidate_count,
            best_candidate_load_latency_ms=best_candidate_effective_load_ms,
            best_candidate_total_latency_ms=best_candidate_total_ms,
            best_candidate_savings_ms=(
                baseline_compute_ms - best_candidate_total_ms
                if best_candidate_count
                else 0.0
            ),
            raw_load_latency_ms=selected_raw_load_ms,
            effective_load_latency_ms=selected_effective_load_ms,
            baseline_recompute_latency_ms=baseline_compute_ms,
            selected_recompute_latency_ms=selected_compute_ms,
            estimated_savings_ms=savings_ms,
            selected_keys=[key for key, _ in selected],
        )
        self._apply_lookup(decision, selected)
        return decision

    def _recompute_latency_ms(self, *, start_token: int, end_token: int) -> float:
        # The quadratic term captures the growing full-attention cost in long
        # contexts.  It is expressed as ms * (end^2 - start^2), making the
        # avoided cost of a retrieved prefix explicit and calibratable.
        start_token = max(0, min(start_token, end_token - 1))
        token_count = max(1, end_token - start_token)
        return (
            token_count * self.config.recompute_ms_per_token
            + (end_token**2 - start_token**2)
            * self.config.recompute_ms_per_token_squared
        )

    def _read_latency_ms(self, blocks: list[tuple[str, str]]) -> float:
        counts = {
            "dram": sum(tier == "dram" for _, tier in blocks),
            "ssd": sum(tier == "ssd" for _, tier in blocks),
        }
        latencies: list[float] = []
        if counts["dram"]:
            latencies.append(
                self.config.dram_read_latency_ms
                + counts["dram"]
                * self.kv_bytes_per_block
                / (self.config.dram_read_bandwidth_gbps * 1_000_000.0)
            )
        if counts["ssd"]:
            latencies.append(
                self.config.ssd_read_latency_ms
                + counts["ssd"]
                * self.kv_bytes_per_block
                / (self.config.ssd_read_bandwidth_gbps * 1_000_000.0)
            )
        if not latencies:
            return 0.0
        read_latency = (
            max(latencies) if self.config.parallel_tier_reads else sum(latencies)
        )
        # SSD hits are promoted into DRAM.  If DRAM is full, the exclusive
        # cache must demote the same number of victims back to SSD.
        promoted_blocks = (
            min(counts["ssd"], self.dram.capacity_blocks)
            if self.dram.capacity_blocks > 0
            else 0
        )
        free_dram_blocks = max(
            0, self.dram.capacity_blocks - len(self.dram.entries)
        )
        demoted_blocks = max(0, promoted_blocks - free_dram_blocks)
        return read_latency + self._write_latency_ms("ssd", demoted_blocks)

    def _write_latency_ms(self, tier: str, blocks: int) -> float:
        if blocks <= 0:
            return 0.0
        if tier == "dram":
            return self.config.dram_write_latency_ms + (
                blocks
                * self.kv_bytes_per_block
                / (self.config.dram_write_bandwidth_gbps * 1_000_000.0)
            )
        return self.config.ssd_write_latency_ms + (
            blocks
            * self.kv_bytes_per_block
            / (self.config.ssd_write_bandwidth_gbps * 1_000_000.0)
        )

    def _apply_lookup(
        self,
        decision: TieredLookupDecision,
        selected: list[tuple[str, str]],
    ) -> None:
        self.metrics.lookup_requests += 1
        self.metrics.cacheable_query_tokens += decision.cacheable_tokens
        self.metrics.hbm_hit_tokens += decision.local_hbm_tokens
        self.metrics.dram_candidate_tokens += (
            decision.available_dram_blocks * self.block_size
        )
        self.metrics.ssd_candidate_tokens += (
            decision.available_ssd_blocks * self.block_size
        )
        self.metrics.recompute_tokens += decision.selected_recompute_tokens
        if decision.decision == "cache":
            self.metrics.cache_decisions += 1
        else:
            self.metrics.recompute_decisions += 1
        self.metrics.dram_hit_tokens += decision.dram_blocks * self.block_size
        self.metrics.ssd_hit_tokens += decision.ssd_blocks * self.block_size
        self.metrics.dram_read_bytes += decision.dram_blocks * self.kv_bytes_per_block
        self.metrics.ssd_read_bytes += decision.ssd_blocks * self.kv_bytes_per_block
        self.metrics.estimated_load_latency_ms += decision.effective_load_latency_ms
        self.metrics.estimated_recompute_avoided_ms += max(
            0.0, decision.estimated_savings_ms
        )
        for key, original_tier in selected:
            sequence = self._next_sequence()
            if original_tier == "dram":
                self.dram.access(key, sequence)
                continue
            if self.dram.capacity_blocks == 0:
                self.ssd.access(key, sequence)
                continue
            entry = self.ssd.remove(key)
            if entry is None:
                continue
            self.metrics.promoted_blocks += 1
            demoted = self.dram.put(entry, sequence)
            for victim in demoted:
                self.metrics.dram_evicted_blocks += 1
                self.metrics.ssd_write_bytes += self.kv_bytes_per_block
                dropped = self.ssd.put(victim, self._next_sequence())
                self.metrics.ssd_evicted_blocks += len(dropped)
        if selected:
            self._persist_ssd_state()

    def store_prefix_blocks(self, block_keys: Iterable[str]) -> TieredStorePlan:
        plan = TieredStorePlan()
        if not self.enabled or self.config.write_policy == "none":
            return plan
        for key in block_keys:
            if self.lookup_tier(key) is not None:
                continue
            entry = TieredBlock(key=key, inserted_at=0, last_accessed_at=0)
            sequence = self._next_sequence()
            if self.dram.capacity_blocks > 0:
                plan.dram_write_blocks += 1
                evicted = self.dram.put(entry, sequence)
                plan.dram_evicted_blocks += len(evicted)
                for victim in evicted:
                    if self.ssd.capacity_blocks > 0:
                        plan.ssd_write_blocks += 1
                    dropped = self.ssd.put(victim, self._next_sequence())
                    plan.ssd_evicted_blocks += len(dropped)
            else:
                if self.ssd.capacity_blocks > 0:
                    plan.ssd_write_blocks += 1
                dropped = self.ssd.put(entry, sequence)
                plan.ssd_evicted_blocks += len(dropped)
            plan.stored_blocks += 1

        plan.raw_store_latency_ms = self._write_latency_ms(
            "dram", plan.dram_write_blocks
        ) + self._write_latency_ms("ssd", plan.ssd_write_blocks)
        plan.effective_store_latency_ms = plan.raw_store_latency_ms * (
            1.0 - self.config.write_overlap_fraction
        )
        self.metrics.stored_blocks += plan.stored_blocks
        self.metrics.dram_write_bytes += (
            plan.dram_write_blocks * self.kv_bytes_per_block
        )
        self.metrics.ssd_write_bytes += plan.ssd_write_blocks * self.kv_bytes_per_block
        self.metrics.dram_evicted_blocks += plan.dram_evicted_blocks
        self.metrics.ssd_evicted_blocks += plan.ssd_evicted_blocks
        self.metrics.estimated_store_latency_ms += plan.effective_store_latency_ms
        if plan.stored_blocks:
            self._persist_ssd_state()
        return plan

    def reset(self) -> None:
        self.dram.clear()
        self.ssd.clear()
        self.metrics = TieredCacheMetrics()
        self._persist_ssd_state()
        self._write_metrics()

    def snapshot(self) -> dict[str, Any]:
        metrics = asdict(self.metrics)
        query_tokens = metrics["cacheable_query_tokens"]
        metrics["total_hit_rate"] = (
            (
                metrics["hbm_hit_tokens"]
                + metrics["dram_hit_tokens"]
                + metrics["ssd_hit_tokens"]
            )
            / query_tokens
            if query_tokens
            else 0.0
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "namespace": self.namespace,
            "block_size": self.block_size,
            "kv_bytes_per_block": self.kv_bytes_per_block,
            "eviction_policy": self.config.eviction_policy,
            "prefetch_policy": self.config.prefetch_policy,
            "dram": {
                "capacity_blocks": self.dram.capacity_blocks,
                "used_blocks": len(self.dram.entries),
            },
            "ssd": {
                "capacity_blocks": self.ssd.capacity_blocks,
                "used_blocks": len(self.ssd.entries),
            },
            "metrics": metrics,
        }

    def write_metrics(self) -> None:
        self._write_metrics()

    def _write_metrics(self) -> None:
        if not self.config.metrics_path:
            return
        path = Path(self.config.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _load_ssd_state(self) -> None:
        if not self.config.ssd_state_path:
            return
        path = Path(self.config.ssd_state_path)
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("instance_id") != self.instance_id:
            raise RuntimeError(
                "SSD cache state instance_id mismatch; cross-Instance KV reuse "
                "is forbidden"
            )
        if payload.get("namespace") != self.namespace:
            raise RuntimeError("SSD cache state model namespace mismatch")
        for item in payload.get("entries", []):
            entry = TieredBlock(**item)
            self._sequence = max(
                self._sequence, entry.inserted_at, entry.last_accessed_at
            )
            dropped = self.ssd.put(entry, self._next_sequence())
            if dropped:
                break

    def _persist_ssd_state(self) -> None:
        if not self.config.ssd_state_path:
            self._write_metrics()
            return
        path = Path(self.config.ssd_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "namespace": self.namespace,
            "entries": [asdict(entry) for entry in self.ssd.entries.values()],
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._write_metrics()
