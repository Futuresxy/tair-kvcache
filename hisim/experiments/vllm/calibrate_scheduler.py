#!/usr/bin/env python3
"""Fit a first scheduler latency profile and replay the exact real workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from hisim.simulation.vllm import (
    LinearLatencyProfile,
    VllmRequestSpec,
    VllmSchedulerSimulator,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_nonnegative_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("at least two paired observations are required")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("prefill observations need at least two computed-token sizes")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return max(0.0, intercept), max(0.0, slope)


def relative_error_percent(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if predicted == 0 else float("inf")
    return abs(predicted - actual) / actual * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--profile-out", required=True, type=Path)
    parser.add_argument("--workload-out", required=True, type=Path)
    parser.add_argument("--simulation-out", required=True, type=Path)
    parser.add_argument("--comparison-out", required=True, type=Path)
    parser.add_argument("--instance-id", default="replica-0")
    parser.add_argument("--num-gpu-blocks", type=int, default=4096)
    args = parser.parse_args()

    real = load_json(args.real)
    requests = real["requests"]
    computed_prompt_tokens = [
        request["prefix_cache_queries"] - request["prefix_cache_hits"]
        for request in requests
    ]
    real_ttfts = [request["ttft_ms"] for request in requests]
    prefill_base_ms, prefill_token_ms = fit_nonnegative_line(
        computed_prompt_tokens, real_ttfts
    )
    real_tpots = [
        request["tpot_ms"] for request in requests if request["tpot_ms"] is not None
    ]
    decode_step_ms = statistics.median(real_tpots)
    profile = LinearLatencyProfile(
        name="rtx4090-qwen3-0.6b-bs1-in-sample-v1",
        scheduler_overhead_ms=0.0,
        prefill_base_ms=prefill_base_ms,
        prefill_token_ms=prefill_token_ms,
        decode_base_ms=decode_step_ms,
        decode_token_ms=0.0,
        decode_context_token_ms=0.0,
        calibrated=True,
    )
    profile_document = {
        "schema_version": 1,
        "latency_profile": profile.__dict__,
        "calibration": {
            "evaluation_kind": "in_sample_fit",
            "source": str(args.real),
            "source_sha256": file_sha256(args.real),
            "accelerator": real["environment"]["cuda_device"],
            "vllm": real["environment"]["vllm"],
            "torch": real["environment"]["torch"],
            "model": real["config"]["model"],
            "batch_size": 1,
            "computed_prompt_tokens": computed_prompt_tokens,
            "ttft_ms": real_ttfts,
            "tpot_ms": real_tpots,
            "limitations": [
                "This is an in-sample fit, not a held-out accuracy result.",
                "The profile currently covers sequential batch-size-1 requests only.",
            ],
        },
    }
    dump_json(args.profile_out, profile_document)

    first_start = requests[0]["start_offset_ms"]
    workload_specs = [
        VllmRequestSpec(
            request_id=request["request_id"],
            prompt_token_ids=request["prompt_token_ids"],
            max_tokens=request["output_tokens"],
            instance_id=args.instance_id,
            arrival_time_ms=request["start_offset_ms"] - first_start,
        )
        for request in requests
    ]
    args.workload_out.parent.mkdir(parents=True, exist_ok=True)
    args.workload_out.write_text(
        "".join(json.dumps(spec.__dict__) + "\n" for spec in workload_specs),
        encoding="utf-8",
    )

    simulator = VllmSchedulerSimulator(
        model=args.model_config,
        instance_id=args.instance_id,
        latency_profile=profile,
        block_size=16,
        num_gpu_blocks=args.num_gpu_blocks,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=16,
    )
    simulated = simulator.run(workload_specs)
    simulated_document = simulated.to_dict()
    dump_json(args.simulation_out, simulated_document)

    comparisons = []
    for real_request, simulated_request in zip(requests, simulated.requests):
        fields = {}
        for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms", "kv_cache_hit_rate"):
            real_value = real_request[field]
            simulated_value = getattr(simulated_request, field)
            fields[field] = {
                "real": real_value,
                "simulated": simulated_value,
                "absolute_error": abs(simulated_value - real_value),
                "absolute_percentage_error": relative_error_percent(
                    simulated_value, real_value
                ),
            }
        comparisons.append({"request_id": real_request["request_id"], **fields})

    real_active_ms = sum(request["e2e_latency_ms"] for request in requests)
    simulated_active_ms = sum(
        request.e2e_latency_ms or 0.0 for request in simulated.requests
    )
    output_tokens = sum(request["output_tokens"] for request in requests)
    comparison = {
        "schema_version": 1,
        "evaluation_kind": "in_sample_fit",
        "real_source": str(args.real),
        "simulation_source": str(args.simulation_out),
        "profile_source": str(args.profile_out),
        "summary": {
            "ttft_mape_percent": statistics.fmean(
                request["ttft_ms"]["absolute_percentage_error"]
                for request in comparisons
            ),
            "tpot_mape_percent": statistics.fmean(
                request["tpot_ms"]["absolute_percentage_error"]
                for request in comparisons
            ),
            "e2e_mape_percent": statistics.fmean(
                request["e2e_latency_ms"]["absolute_percentage_error"]
                for request in comparisons
            ),
            "kv_cache_hit_rate_absolute_error": abs(
                simulated.kv_cache_hit_rate - real["summary"]["kv_cache_hit_rate"]
            ),
            "real_active_output_throughput_tokens_per_s": output_tokens
            / (real_active_ms / 1000.0),
            "simulated_active_output_throughput_tokens_per_s": output_tokens
            / (simulated_active_ms / 1000.0),
        },
        "requests": comparisons,
    }
    comparison["summary"]["active_throughput_error_percent"] = (
        relative_error_percent(
            comparison["summary"]["simulated_active_output_throughput_tokens_per_s"],
            comparison["summary"]["real_active_output_throughput_tokens_per_s"],
        )
    )
    dump_json(args.comparison_out, comparison)
    print(json.dumps(comparison["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
