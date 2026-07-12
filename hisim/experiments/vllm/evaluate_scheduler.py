#!/usr/bin/env python3
"""Evaluate a fixed scheduler profile on a held-out real vLLM workload."""

from __future__ import annotations

import argparse
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


def relative_error_percent(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if predicted == 0 else float("inf")
    return abs(predicted - actual) / actual * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--simulation-out", required=True, type=Path)
    parser.add_argument("--comparison-out", required=True, type=Path)
    parser.add_argument("--instance-id", default="replica-0")
    args = parser.parse_args()

    real = load_json(args.real)
    profile_document = load_json(args.profile)
    profile = LinearLatencyProfile(**profile_document["latency_profile"])
    real_requests = real["requests"]
    first_start = real_requests[0]["start_offset_ms"]
    workload = [
        VllmRequestSpec(
            request_id=request["request_id"],
            prompt_token_ids=request["prompt_token_ids"],
            max_tokens=request["output_tokens"],
            instance_id=args.instance_id,
            arrival_time_ms=request["start_offset_ms"] - first_start,
        )
        for request in real_requests
    ]
    simulator = VllmSchedulerSimulator(
        model=args.model_config,
        instance_id=args.instance_id,
        latency_profile=profile,
        block_size=16,
        num_gpu_blocks=4096,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=16,
    )
    simulated = simulator.run(workload)
    dump_json(args.simulation_out, simulated.to_dict())

    comparisons = []
    for real_request, simulated_request in zip(real_requests, simulated.requests):
        fields = {}
        for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms", "kv_cache_hit_rate"):
            actual = real_request[field]
            predicted = getattr(simulated_request, field)
            fields[field] = {
                "real": actual,
                "simulated": predicted,
                "absolute_error": abs(predicted - actual),
                "absolute_percentage_error": relative_error_percent(
                    predicted, actual
                ),
            }
        comparisons.append({"request_id": real_request["request_id"], **fields})

    real_active_ms = sum(request["e2e_latency_ms"] for request in real_requests)
    simulated_active_ms = sum(
        request.e2e_latency_ms or 0.0 for request in simulated.requests
    )
    output_tokens = sum(request["output_tokens"] for request in real_requests)
    summary = {
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
    }
    summary["active_throughput_error_percent"] = relative_error_percent(
        summary["simulated_active_output_throughput_tokens_per_s"],
        summary["real_active_output_throughput_tokens_per_s"],
    )
    comparison = {
        "schema_version": 1,
        "evaluation_kind": "held_out",
        "real_source": str(args.real),
        "simulation_source": str(args.simulation_out),
        "profile_source": str(args.profile),
        "summary": summary,
        "requests": comparisons,
    }
    dump_json(args.comparison_out, comparison)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
