#!/usr/bin/env python3
"""Fit an RTX 4090 latency profile from multiple sequential workload grids."""

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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="rtx4090-qwen3-0.6b-bs1-grid-v2")
    args = parser.parse_args()

    documents = [load_json(path) for path in args.real]
    requests = [request for document in documents for request in document["requests"]]
    computed_prompt_tokens = [
        request["prefix_cache_queries"] - request["prefix_cache_hits"]
        for request in requests
    ]
    ttfts = [request["ttft_ms"] for request in requests]
    prefill_base_ms, prefill_token_ms = fit_nonnegative_line(
        computed_prompt_tokens, ttfts
    )
    decode_context_tokens = [
        request["prompt_tokens"] + (request["output_tokens"] - 1) / 2.0
        for request in requests
        if request["tpot_ms"] is not None
    ]
    tpots = [request["tpot_ms"] for request in requests if request["tpot_ms"] is not None]
    decode_base_ms, decode_context_token_ms = fit_nonnegative_line(
        decode_context_tokens, tpots
    )
    profile = LinearLatencyProfile(
        name=args.name,
        scheduler_overhead_ms=0.0,
        prefill_base_ms=prefill_base_ms,
        prefill_token_ms=prefill_token_ms,
        decode_base_ms=decode_base_ms,
        decode_token_ms=0.0,
        decode_context_token_ms=decode_context_token_ms,
        calibrated=True,
    )
    output = {
        "schema_version": 2,
        "latency_profile": profile.__dict__,
        "calibration": {
            "evaluation_kind": "multi_length_fit",
            "sources": [
                {"path": str(path), "sha256": file_sha256(path)} for path in args.real
            ],
            "accelerator": documents[0]["environment"]["cuda_device"],
            "vllm": documents[0]["environment"]["vllm"],
            "torch": documents[0]["environment"]["torch"],
            "model": documents[0]["config"]["model"],
            "batch_size": 1,
            "request_count": len(requests),
            "computed_prompt_tokens": computed_prompt_tokens,
            "decode_context_tokens": decode_context_tokens,
            "ttft_ms": ttfts,
            "tpot_ms": tpots,
            "limitations": [
                "The profile covers sequential batch-size-1 requests.",
                "Concurrent batch-shape validation is a separate experiment.",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["latency_profile"], ensure_ascii=False))


if __name__ == "__main__":
    main()
