#!/usr/bin/env python3
"""Validate cold/warm tiered-cache reuse through SGLang's real Engine path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--simulation-config", required=True)
    parser.add_argument("--tiered-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HISIM_CONFIG_PATH"] = str(Path(args.simulation_config).resolve())
    os.environ["HISIM_SGLANG_TIERED_KV_CONFIG"] = str(
        Path(args.tiered_config).resolve()
    )
    tiered_payload = json.loads(Path(args.tiered_config).read_text(encoding="utf-8"))
    tiered_settings = tiered_payload.get("tiered_kv_cache", tiered_payload)
    os.environ["HISIM_SGLANG_INSTANCE_ID"] = tiered_settings["instance_id"]
    os.environ.setdefault("HISIM_SIMULATION_MODE", "OFFLINE")
    os.environ.setdefault("HISIM_OUTPUT_DIR", "/tmp/hisim/sglang-tiered-validation")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    # Hooks read the environment during Scheduler construction, so framework
    # imports intentionally occur only after all namespace/config values exist.
    from hisim.dataset import GenericRequest, SimpleDataset
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner
    from hisim.simulation.types import BenchmarkConfig
    from sglang.srt.server_args import ServerArgs

    prompt = [1000 + (index % 1000) for index in range(args.prompt_tokens)]
    dataset = SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=prompt,
                input_length=len(prompt),
                output_length=args.output_tokens,
            )
        ]
    )
    runner = SGLangBenchmarkRunner(
        ServerArgs(
            model_path=str(Path(args.model).resolve()),
            load_format="dummy",
            device="cpu",
            max_total_tokens=args.max_total_tokens,
            page_size=1,
            enable_hierarchical_cache=True,
            hicache_ratio=2.0,
            hicache_write_policy="write_through",
            hicache_storage_backend="file",
            hicache_storage_backend_extra_config=json.dumps(
                {"prefetch_threshold": 1}
            ),
        )
    )
    try:
        benchmark_config = BenchmarkConfig(ignore_request_timestamp=True)
        cold = runner.benchmark(benchmark_config, dataset=dataset)
        runner.flush_cache()
        warm = runner.benchmark(benchmark_config, dataset=dataset)
    finally:
        runner.shutdown()

    result = {
        "schema_version": 1,
        "framework": "sglang",
        "framework_version": "0.5.6.post2",
        "model": str(Path(args.model).resolve()),
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "cold": cold,
        "warm": warm,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
