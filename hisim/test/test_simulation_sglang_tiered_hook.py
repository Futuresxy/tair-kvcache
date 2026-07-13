from types import SimpleNamespace
from typing import NamedTuple

from hisim.simulation.sglang.sglang_hook import C_HiRadixCacheHook


class _MatchResult(NamedTuple):
    device_indices: list[int]
    last_device_node: object
    last_host_node: object
    host_hit_length: int


class _FakeHiRadixCache:
    def check_hicache_events(self):
        return None

    def match_prefix(self):
        return self.match_result

    def reset(self):
        return None


def test_native_host_pool_is_staging_when_shared_tier_policy_is_enabled():
    C_HiRadixCacheHook.hook(_FakeHiRadixCache)
    cache = object.__new__(_FakeHiRadixCache)
    device_node = object()
    host_node = object()
    cache.match_result = _MatchResult([], device_node, host_node, 8)
    cache.cache_controller = SimpleNamespace(
        storage_backend=SimpleNamespace(lookup_tier=lambda _key: "dram")
    )

    result = cache.match_prefix()

    assert result.host_hit_length == 0
    assert result.last_host_node is device_node
