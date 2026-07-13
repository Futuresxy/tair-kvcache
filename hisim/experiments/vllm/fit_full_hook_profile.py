#!/usr/bin/env python3
"""Fit Worker sleep time after subtracting full-engine fast-forward overhead."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from hisim.simulation.vllm import LinearLatencyProfile


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def fit_nonnegative_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    if slope < 0:
        return mean_y, 0.0
    return max(0.0, mean_y - slope * mean_x), slope


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, action="append", type=Path)
    parser.add_argument("--overhead", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.real) != len(args.overhead):
        raise ValueError("each --real needs one matching --overhead report")

    computed_prompt_tokens = []
    decode_context_tokens = []
    execution_ttfts = []
    execution_tpots = []
    source_pairs = []
    for real_path, overhead_path in zip(args.real, args.overhead):
        real = load_json(real_path)
        overhead = load_json(overhead_path)
        if len(real["requests"]) != len(overhead["requests"]):
            raise ValueError(f"request count mismatch: {real_path}")
        for real_request, overhead_request in zip(
            real["requests"], overhead["requests"]
        ):
            if real_request["prompt_token_ids"] != overhead_request["prompt_token_ids"]:
                raise ValueError(f"prompt token mismatch: {real_request['request_id']}")
            computed_prompt_tokens.append(
                real_request["prefix_cache_queries"]
                - real_request["prefix_cache_hits"]
            )
            execution_ttfts.append(
                max(0.0, real_request["ttft_ms"] - overhead_request["ttft_ms"])
            )
            decode_context_tokens.append(
                real_request["prompt_tokens"]
                + (real_request["output_tokens"] - 1) / 2.0
            )
            execution_tpots.append(
                max(0.0, real_request["tpot_ms"] - overhead_request["tpot_ms"])
            )
        source_pairs.append(
            {
                "real": str(real_path),
                "real_sha256": sha256(real_path),
                "overhead": str(overhead_path),
                "overhead_sha256": sha256(overhead_path),
            }
        )

    prefill_base_ms, prefill_token_ms = fit_nonnegative_line(
        computed_prompt_tokens, execution_ttfts
    )
    decode_base_ms, decode_context_token_ms = fit_nonnegative_line(
        decode_context_tokens, execution_tpots
    )
    profile = LinearLatencyProfile(
        name="rtx4090-qwen3-0.6b-full-engine-worker-v4",
        scheduler_overhead_ms=0.0,
        prefill_base_ms=prefill_base_ms,
        prefill_token_ms=prefill_token_ms,
        decode_base_ms=decode_base_ms,
        decode_token_ms=0.0,
        decode_context_token_ms=decode_context_token_ms,
        calibrated=True,
    )
    output = {
        "schema_version": 1,
        "latency_profile": profile.__dict__,
        "calibration": {
            "evaluation_kind": "full_engine_overhead_subtracted_fit",
            "source_pairs": source_pairs,
            "computed_prompt_tokens": computed_prompt_tokens,
            "execution_ttft_ms": execution_ttfts,
            "decode_context_tokens": decode_context_tokens,
            "execution_tpot_ms": execution_tpots,
            "request_count": len(execution_ttfts),
            "batch_size": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["latency_profile"], ensure_ascii=False))


if __name__ == "__main__":
    main()
