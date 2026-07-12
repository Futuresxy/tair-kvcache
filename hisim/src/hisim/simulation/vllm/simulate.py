"""Command-line entry point for the vLLM scheduler simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .scheduler_simulator import (
    LinearLatencyProfile,
    VllmRequestSpec,
    VllmSchedulerSimulator,
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_workload(path: Path, default_instance_id: str) -> list[VllmRequestSpec]:
    requests = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                payload.setdefault("instance_id", default_instance_id)
                requests.append(VllmRequestSpec(**payload))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid workload line {line_number}: {error}") from error
    return requests


def load_latency_profile(path: Path | None) -> LinearLatencyProfile:
    if path is None:
        return LinearLatencyProfile()
    payload = _load_json(path)
    if "latency_profile" in payload:
        payload = payload["latency_profile"]
    try:
        return LinearLatencyProfile(**payload)
    except TypeError as error:
        raise ValueError(f"invalid latency profile {path}: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HiSim with the real vLLM 0.23 scheduler and a mock model runner"
    )
    parser.add_argument("--model", required=True, help="local model/config directory")
    parser.add_argument("--workload", required=True, type=Path, help="JSONL workload")
    parser.add_argument("--output", required=True, type=Path, help="result JSON")
    parser.add_argument("--instance-id", default="default", help="KV-cache namespace")
    parser.add_argument("--latency-profile", type=Path, help="latency profile JSON")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--allow-unsupported-vllm",
        action="store_true",
        help="disable the exact vLLM 0.23.0 version guard",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    requests = load_workload(args.workload, args.instance_id)
    simulator = VllmSchedulerSimulator(
        model=args.model,
        instance_id=args.instance_id,
        latency_profile=load_latency_profile(args.latency_profile),
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        strict_version=not args.allow_unsupported_vllm,
    )
    result = simulator.run(requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict()["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
