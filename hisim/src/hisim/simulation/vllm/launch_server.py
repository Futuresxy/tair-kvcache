"""Launch the complete vLLM OpenAI server with HiSim model execution."""

from __future__ import annotations

import argparse
import os
import sys


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
    return parser.parse_known_args(argv)


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
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
