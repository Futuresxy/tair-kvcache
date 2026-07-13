"""vLLM-backed scheduling simulation for HiSim.

The implementation intentionally imports vLLM lazily.  This keeps the existing
SGLang environment usable while the vLLM 0.23 integration lives in its isolated
``hisim-vllm023`` environment.
"""

from .scheduler_simulator import (
    COMPATIBLE_VLLM_VERSIONS,
    LinearLatencyProfile,
    RequestMetrics,
    SimulationResult,
    VllmRequestSpec,
    VllmSchedulerSimulator,
)
from .benchmark_runner import VllmBenchmarkRunner
from .engine import create_hisim_llm
from .tiered_cache import TieredKVCache, TieredKVCacheConfig

__all__ = [
    "COMPATIBLE_VLLM_VERSIONS",
    "LinearLatencyProfile",
    "RequestMetrics",
    "SimulationResult",
    "VllmRequestSpec",
    "VllmBenchmarkRunner",
    "create_hisim_llm",
    "TieredKVCache",
    "TieredKVCacheConfig",
    "VllmSchedulerSimulator",
]
