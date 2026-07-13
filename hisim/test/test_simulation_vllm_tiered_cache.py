import json

import pytest

from hisim.simulation.vllm import TieredKVCache, TieredKVCacheConfig


def _cache(tmp_path, *, instance_id="instance-a", **overrides):
    config = TieredKVCacheConfig.from_mapping(
        {
            "dram_capacity_blocks": 2,
            "ssd_capacity_blocks": 8,
            "prefetch_policy": "always",
            **overrides,
        }
    )
    return TieredKVCache(
        instance_id=instance_id,
        namespace=f"model:test:{instance_id}",
        block_size=2,
        kv_bytes_per_block=1024,
        config=config,
    )


@pytest.mark.parametrize(
    ("policy", "expected_demoted_index"),
    [("lru", 1), ("fifo", 0)],
)
def test_tiered_cache_lru_and_fifo_demote_to_ssd(
    tmp_path, policy, expected_demoted_index
):
    cache = _cache(tmp_path, eviction_policy=policy)
    token_ids = [1, 2, 3, 4, 5, 6]
    keys = cache.prefix_block_keys(token_ids)
    cache.store_prefix_blocks(keys[:2])

    # LRU refreshes block zero; FIFO intentionally ignores that access.
    decision = cache.plan_lookup(
        request_id="touch-first",
        prompt_token_ids=[1, 2, 99],
        local_hbm_tokens=0,
    )
    assert decision.selected_external_tokens == 2
    cache.store_prefix_blocks(keys[2:])

    assert cache.lookup_tier(keys[expected_demoted_index]) == "ssd"
    assert len(cache.dram.entries) == 2
    assert len(cache.ssd.entries) == 1


def test_cost_aware_policy_selects_cache_only_when_faster(tmp_path):
    prompt = list(range(33))
    fast = _cache(
        tmp_path,
        dram_capacity_blocks=20,
        recompute_ms_per_token=1.0,
        prefetch_policy="cost_aware",
    )
    fast.store_prefix_blocks(fast.prefix_block_keys(prompt, upto_tokens=32))
    cache_decision = fast.plan_lookup(
        request_id="fast-dram", prompt_token_ids=prompt, local_hbm_tokens=0
    )

    slow = _cache(
        tmp_path,
        dram_capacity_blocks=0,
        ssd_capacity_blocks=20,
    )
    # Replace the helper's small-block cache with a deliberately slow SSD.
    slow.config.prefetch_policy = "cost_aware"
    slow.config.recompute_ms_per_token = 0.01
    slow.config.ssd_read_bandwidth_gbps = 0.001
    slow.config.ssd_read_latency_ms = 100.0
    slow.store_prefix_blocks(slow.prefix_block_keys(prompt, upto_tokens=32))
    recompute_decision = slow.plan_lookup(
        request_id="slow-ssd", prompt_token_ids=prompt, local_hbm_tokens=0
    )

    assert cache_decision.decision == "cache"
    assert cache_decision.selected_external_tokens == 32
    assert cache_decision.estimated_savings_ms > 0
    assert recompute_decision.decision == "recompute"
    assert recompute_decision.selected_external_tokens == 0


def test_ssd_state_is_persistent_and_instance_isolated(tmp_path):
    state_path = tmp_path / "ssd-state.json"
    first = _cache(
        tmp_path,
        dram_capacity_blocks=0,
        ssd_capacity_blocks=4,
        ssd_state_path=str(state_path),
    )
    keys = first.prefix_block_keys([1, 2, 3, 4])
    first.store_prefix_blocks(keys)
    assert json.loads(state_path.read_text())["instance_id"] == "instance-a"

    restored = _cache(
        tmp_path,
        dram_capacity_blocks=0,
        ssd_capacity_blocks=4,
        ssd_state_path=str(state_path),
    )
    assert all(restored.lookup_tier(key) == "ssd" for key in keys)

    with pytest.raises(RuntimeError, match="cross-Instance"):
        _cache(
            tmp_path,
            instance_id="instance-b",
            dram_capacity_blocks=0,
            ssd_capacity_blocks=4,
            ssd_state_path=str(state_path),
        )
