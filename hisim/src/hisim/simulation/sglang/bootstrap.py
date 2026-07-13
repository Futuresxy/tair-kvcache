"""Install HiSim hooks in both parent and spawned SGLang processes."""

from __future__ import annotations

import os
from typing import Any


_HOOKS_INSTALLED = False


def install_sglang_simulation_hooks() -> None:
    """Install import-time hooks before SGLang runtime classes are defined."""

    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    import torch

    import hisim.hook as hisim_hook
    from hisim.simulation.sglang import sgl_kernel_hook, sglang_hook

    if not torch.cuda.is_available():
        hisim_hook.install_module_hooks([sgl_kernel_hook.M_SGLangKernelLoadUtilHook])
    hisim_hook.install_class_hooks(
        [
            sglang_hook.C_EngineHook,
            sglang_hook.C_SchedulerHook,
            sglang_hook.C_ModelRunnerHook,
            sglang_hook.C_TokenizerManagerHook,
            sglang_hook.C_StorageBackendFactory,
            sglang_hook.C_HiCacheController,
            sglang_hook.C_HiRadixCacheHook,
        ]
    )
    _HOOKS_INSTALLED = True


def run_hisim_scheduler_process(*args: Any, **kwargs: Any) -> Any:
    """Spawn-safe SGLang scheduler target that installs hooks in the child."""

    install_sglang_simulation_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_hisim_data_parallel_controller_process(*args: Any, **kwargs: Any) -> Any:
    """Spawn-safe DP controller target; its Scheduler imports are hooked too."""

    install_sglang_simulation_hooks()
    from sglang.srt.managers.data_parallel_controller import (
        run_data_parallel_controller_process,
    )

    return run_data_parallel_controller_process(*args, **kwargs)


def patch_sglang_process_entrypoints() -> None:
    """Make Engine pickle HiSim child targets instead of raw SGLang targets."""

    install_sglang_simulation_hooks()
    import sglang.srt.entrypoints.engine as engine_module

    engine_module.run_scheduler_process = run_hisim_scheduler_process
    engine_module.run_data_parallel_controller_process = (
        run_hisim_data_parallel_controller_process
    )
    os.environ["HISIM_SGLANG_CHILD_HOOKS"] = "1"


__all__ = [
    "install_sglang_simulation_hooks",
    "patch_sglang_process_entrypoints",
    "run_hisim_data_parallel_controller_process",
    "run_hisim_scheduler_process",
]
