import json
from types import SimpleNamespace

import pytest

from hisim.simulation.sglang.tiered_adapter import (
    SGLangTieredCacheBackend,
    SGLangTieredCacheSettings,
    load_sglang_tiered_cache_settings,
)
from hisim.simulation.tiered_cache import TieredKVCacheConfig


class _HostPool:
    page_size = 2
    size_per_token = 512
    dtype = "float16"


def _backend(**overrides):
    config = TieredKVCacheConfig.from_mapping(
        {
            "dram_capacity_blocks": 2,
            "ssd_capacity_blocks": 8,
            "prefetch_policy": "always",
            **overrides,
        }
    )
    return SGLangTieredCacheBackend(
        settings=SGLangTieredCacheSettings(
            instance_id="sglang-instance-a",
            config=config,
        ),
        storage_config=SimpleNamespace(
            model_name="Qwen3-0.6B", tp_rank=0, tp_size=1
        ),
        mem_pool_host=_HostPool(),
    )


def test_sglang_adapter_uses_shared_dram_ssd_policy_and_metrics():
    backend = _backend()
    keys = [f"page-{index}" for index in range(4)]
    assert backend.batch_set(keys, values=[None] * len(keys))
    assert backend.cache.lookup_tier(keys[0]) == "ssd"
    assert backend.cache.lookup_tier(keys[3]) == "dram"

    backend.register_request_context(
        "request-a",
        local_hbm_tokens=2,
        local_dram_tokens=0,
        prompt_tokens=9,
    )
    decision = backend.plan_prefetch("request-a", keys[:3])

    assert decision.decision == "cache"
    assert decision.local_hbm_tokens == 2
    assert decision.ssd_blocks == 2
    assert decision.dram_blocks == 1
    assert decision.selected_external_tokens == 6
    snapshot = backend.snapshot()
    assert snapshot["framework"] == "sglang"
    assert snapshot["metrics"]["hbm_hit_tokens"] == 2
    assert snapshot["metrics"]["dram_hit_tokens"] == 2
    assert snapshot["metrics"]["ssd_hit_tokens"] == 4
    assert snapshot["metrics"]["total_hit_rate"] == pytest.approx(1.0)

    # SGLang prefetch is best-effort: policy selection and runtime completion
    # are intentionally distinct, and only completed pages are promoted.
    backend.record_prefetch_completion("request-a", 4)
    snapshot = backend.snapshot()
    assert snapshot["runtime_metrics"]["planned_prefetch_tokens"] == 6
    assert snapshot["runtime_metrics"]["completed_prefetch_tokens"] == 4
    assert snapshot["runtime_metrics"]["ssd_hit_tokens"] == 4
    assert snapshot["runtime_metrics"]["prefetch_completion_rate"] == pytest.approx(
        2 / 3
    )


def test_sglang_adapter_cost_aware_policy_can_choose_recompute():
    backend = _backend(
        dram_capacity_blocks=0,
        ssd_capacity_blocks=4,
        prefetch_policy="cost_aware",
        recompute_ms_per_token=0.01,
        ssd_read_bandwidth_gbps=0.001,
        ssd_read_latency_ms=100.0,
    )
    keys = ["slow-0", "slow-1", "slow-2"]
    backend.batch_set(keys, values=[None] * len(keys))
    backend.register_request_context(
        "request-slow",
        local_hbm_tokens=0,
        local_dram_tokens=0,
        prompt_tokens=7,
    )

    decision = backend.plan_prefetch("request-slow", keys)

    assert decision.decision == "recompute"
    assert decision.selected_external_tokens == 0
    assert decision.best_candidate_blocks > 0
    assert decision.best_candidate_savings_ms < 0
    assert backend.batch_exists(keys) == 3


def test_sglang_adapter_uses_real_model_kv_bytes_when_mock_pool_is_compact():
    host_pool = _HostPool()
    host_pool.device_pool = SimpleNamespace(hisim_kv_bytes_per_token=114688)
    backend = SGLangTieredCacheBackend(
        settings=SGLangTieredCacheSettings(
            instance_id="sglang-instance-bytes",
            config=TieredKVCacheConfig.from_mapping(
                {"dram_capacity_blocks": 1}
            ),
        ),
        storage_config=SimpleNamespace(model_name="Qwen3-0.6B"),
        mem_pool_host=host_pool,
    )

    assert backend.kv_bytes_per_block == 229376


def test_sglang_settings_require_and_preserve_instance_isolation(
    tmp_path, monkeypatch
):
    path = tmp_path / "simulation.json"
    path.write_text(
        json.dumps(
            {
                "platform": {
                    "memory_read_bandwidth_gb": 11,
                    "disk_read_bandwidth_gb": 3,
                },
                "tiered_kv_cache": {
                    "enabled": True,
                    "dram_capacity_blocks": 2,
                    "ssd_capacity_blocks": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HISIM_SGLANG_TIERED_KV_CONFIG", raising=False)
    monkeypatch.delenv("HISIM_SGLANG_INSTANCE_ID", raising=False)

    with pytest.raises(RuntimeError, match="Instance isolation"):
        load_sglang_tiered_cache_settings(path)

    monkeypatch.setenv("HISIM_SGLANG_INSTANCE_ID", "replica-0")
    settings = load_sglang_tiered_cache_settings(path)
    assert settings is not None
    assert settings.instance_id == "replica-0"
    assert settings.config.dram_read_bandwidth_gbps == 11
    assert settings.config.ssd_read_bandwidth_gbps == 3

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tiered_kv_cache"]["instance_id"] = "replica-1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must equal"):
        load_sglang_tiered_cache_settings(path)
