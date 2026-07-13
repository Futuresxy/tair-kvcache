"""Launch the complete vLLM OpenAI server with HiSim model execution."""

from __future__ import annotations

import argparse
import os
import sys

from .tiered_cache import TieredKVCacheConfig


WORKER_CLASS = "hisim.simulation.vllm.worker.HiSimWorker"


def _extract_hisim_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hisim-instance-id", required=True)
    parser.add_argument("--hisim-latency-profile")
    parser.add_argument(
        "--hisim-execution-mode",
        choices=("wall_clock", "fast_forward"),
        default="wall_clock",
    )
    parser.add_argument("--hisim-time-scale", type=float, default=1.0)
    parser.add_argument("--hisim-trace-path")
    parser.add_argument("--hisim-num-gpu-blocks", type=int, default=4096)
    parser.add_argument("--hisim-token-id", type=int, default=9707)
    parser.add_argument("--hisim-tiered-kv-config")
    parser.add_argument("--hisim-dram-capacity-gb", type=float)
    parser.add_argument("--hisim-ssd-capacity-gb", type=float)
    parser.add_argument("--hisim-dram-capacity-blocks", type=int)
    parser.add_argument("--hisim-ssd-capacity-blocks", type=int)
    parser.add_argument("--hisim-cache-policy", choices=("lru", "fifo"))
    parser.add_argument(
        "--hisim-prefetch-policy",
        choices=("none", "always", "cost_aware"),
    )
    parser.add_argument("--hisim-recompute-ms-per-token", type=float)
    parser.add_argument("--hisim-recompute-ms-per-token-squared", type=float)
    parser.add_argument("--hisim-tiered-metrics-path")
    parser.add_argument("--hisim-ssd-state-path")
    return parser.parse_known_args(argv)


def _build_tiered_config(
    args: argparse.Namespace,
) -> TieredKVCacheConfig | None:
    direct_tier_options = (
        args.hisim_dram_capacity_gb,
        args.hisim_ssd_capacity_gb,
        args.hisim_dram_capacity_blocks,
        args.hisim_ssd_capacity_blocks,
    )
    if args.hisim_tiered_kv_config:
        config = TieredKVCacheConfig.from_json_file(args.hisim_tiered_kv_config)
    elif any(value is not None for value in direct_tier_options):
        config = TieredKVCacheConfig()
    else:
        return None

    if args.hisim_dram_capacity_gb is not None:
        config.dram_capacity_bytes = int(args.hisim_dram_capacity_gb * 10**9)
    if args.hisim_ssd_capacity_gb is not None:
        config.ssd_capacity_bytes = int(args.hisim_ssd_capacity_gb * 10**9)
    if args.hisim_dram_capacity_blocks is not None:
        config.dram_capacity_blocks = args.hisim_dram_capacity_blocks
    if args.hisim_ssd_capacity_blocks is not None:
        config.ssd_capacity_blocks = args.hisim_ssd_capacity_blocks
    if args.hisim_cache_policy is not None:
        config.eviction_policy = args.hisim_cache_policy
    if args.hisim_prefetch_policy is not None:
        config.prefetch_policy = args.hisim_prefetch_policy
    if args.hisim_recompute_ms_per_token is not None:
        config.recompute_ms_per_token = args.hisim_recompute_ms_per_token
    if args.hisim_recompute_ms_per_token_squared is not None:
        config.recompute_ms_per_token_squared = (
            args.hisim_recompute_ms_per_token_squared
        )
    if args.hisim_tiered_metrics_path is not None:
        config.metrics_path = args.hisim_tiered_metrics_path
    if args.hisim_ssd_state_path is not None:
        config.ssd_state_path = args.hisim_ssd_state_path
    config.validate()
    return config


def _set_worker_environment(args: argparse.Namespace) -> None:
    os.environ["HISIM_VLLM_INSTANCE_ID"] = args.hisim_instance_id
    os.environ["HISIM_VLLM_EXECUTION_MODE"] = args.hisim_execution_mode
    os.environ["HISIM_VLLM_TIME_SCALE"] = str(args.hisim_time_scale)
    os.environ["HISIM_VLLM_NUM_GPU_BLOCKS"] = str(args.hisim_num_gpu_blocks)
    os.environ["HISIM_VLLM_TOKEN_ID"] = str(args.hisim_token_id)
    if args.hisim_latency_profile:
        os.environ["HISIM_VLLM_LATENCY_PROFILE"] = args.hisim_latency_profile
    if args.hisim_trace_path:
        os.environ["HISIM_VLLM_TRACE_PATH"] = args.hisim_trace_path

    # The HiSim worker performs configuration only and needs no accelerator.
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("PYTHONHASHSEED", "0")


def main(argv: list[str] | None = None) -> None:
    hisim_args, vllm_argv = _extract_hisim_args(argv or sys.argv[1:])
    _set_worker_environment(hisim_args)

    # Import vLLM only after setting VLLM_TARGET_DEVICE.
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.config import KVTransferConfig

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI server with HiSim model execution"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(vllm_argv)
    if args.model_tag is not None:
        args.model = args.model_tag
    args.worker_cls = WORKER_CLASS
    args.distributed_executor_backend = "uni"
    args.enforce_eager = True
    args.enable_prefix_caching = True
    args.num_gpu_blocks_override = hisim_args.hisim_num_gpu_blocks
    args.async_scheduling = False
    tiered_config = _build_tiered_config(hisim_args)
    if tiered_config is not None:
        args.kv_transfer_config = KVTransferConfig(
            kv_connector="HiSimTieredKVConnector",
            kv_connector_module_path=(
                "hisim.simulation.vllm.tiered_connector"
            ),
            kv_role="kv_both",
            kv_connector_extra_config={
                "instance_id": hisim_args.hisim_instance_id,
                "tiered_config": vars(tiered_config),
            },
            kv_load_failure_policy="recompute",
        )
        args.disable_hybrid_kv_cache_manager = True
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
