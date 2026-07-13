"""Factories for complete vLLM offline engines backed by HiSimWorker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


WORKER_CLASS = "hisim.simulation.vllm.worker.HiSimWorker"


def create_hisim_llm(
    *,
    model: str,
    instance_id: str,
    latency_profile_path: str | Path | None = None,
    execution_mode: str = "fast_forward",
    time_scale: float = 1.0,
    trace_path: str | Path | None = None,
    num_gpu_blocks: int = 4096,
    generated_token_id: int = 9707,
    **llm_kwargs: Any,
):
    """Create a standard ``vllm.LLM`` whose model worker is replaced by HiSim.

    The returned object is vLLM's native offline API, not a lookalike wrapper.
    Tokenization, EngineCore, scheduling, output processing and prefix caching
    all execute through vLLM 0.23.
    """
    if not instance_id:
        raise ValueError("instance_id must not be empty")
    if execution_mode not in {"wall_clock", "fast_forward"}:
        raise ValueError("execution_mode must be wall_clock or fast_forward")
    if time_scale <= 0 or num_gpu_blocks <= 0:
        raise ValueError("time_scale and num_gpu_blocks must be positive")

    os.environ["HISIM_VLLM_INSTANCE_ID"] = instance_id
    os.environ["HISIM_VLLM_EXECUTION_MODE"] = execution_mode
    os.environ["HISIM_VLLM_TIME_SCALE"] = str(time_scale)
    os.environ["HISIM_VLLM_NUM_GPU_BLOCKS"] = str(num_gpu_blocks)
    os.environ["HISIM_VLLM_TOKEN_ID"] = str(generated_token_id)
    if latency_profile_path is not None:
        os.environ["HISIM_VLLM_LATENCY_PROFILE"] = str(latency_profile_path)
    else:
        os.environ.pop("HISIM_VLLM_LATENCY_PROFILE", None)
    if trace_path is not None:
        os.environ["HISIM_VLLM_TRACE_PATH"] = str(trace_path)
    else:
        os.environ.pop("HISIM_VLLM_TRACE_PATH", None)
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    from vllm import LLM

    options = {
        "worker_cls": WORKER_CLASS,
        "distributed_executor_backend": "uni",
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "num_gpu_blocks_override": num_gpu_blocks,
        "block_size": 16,
        "async_scheduling": False,
        **llm_kwargs,
    }
    return LLM(model=model, **options)
