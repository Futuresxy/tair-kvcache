"""Compatibility exports for the framework-neutral tiered-cache model.

New code should import from :mod:`hisim.simulation.tiered_cache`.  Keeping this
module avoids breaking existing vLLM integrations and experiment scripts.
"""

from hisim.simulation.tiered_cache import (
    TieredBlock,
    TieredCacheMetrics,
    TieredKVCache,
    TieredKVCacheConfig,
    TieredLookupDecision,
    TieredStorePlan,
)

__all__ = [
    "TieredBlock",
    "TieredCacheMetrics",
    "TieredKVCache",
    "TieredKVCacheConfig",
    "TieredLookupDecision",
    "TieredStorePlan",
]
