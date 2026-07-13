#!/usr/bin/env python3
"""Compare real vLLM and full HiSim-worker server reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def relative_error_percent(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if predicted == 0 else float("inf")
    return abs(predicted - actual) / actual * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--hisim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    real = load_json(args.real)
    hisim = load_json(args.hisim)
    if len(real["requests"]) != len(hisim["requests"]):
        raise ValueError("request count mismatch")

    comparisons = []
    for real_request, hisim_request in zip(real["requests"], hisim["requests"]):
        if real_request["prompt_token_ids"] != hisim_request["prompt_token_ids"]:
            raise ValueError(f"prompt token mismatch for {real_request['request_id']}")
        fields = {}
        for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms", "kv_cache_hit_rate"):
            actual = real_request[field]
            predicted = hisim_request[field]
            fields[field] = {
                "real": actual,
                "hisim_full_hook": predicted,
                "absolute_error": abs(predicted - actual),
                "absolute_percentage_error": relative_error_percent(
                    predicted, actual
                ),
            }
        comparisons.append({"request_id": real_request["request_id"], **fields})

    summary = {}
    for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms"):
        summary[f"{field.removesuffix('_ms')}_mape_percent"] = statistics.fmean(
            request[field]["absolute_percentage_error"] for request in comparisons
        )
    real_throughput = real["summary"]["active_output_throughput_tokens_per_s"]
    hisim_throughput = hisim["summary"]["active_output_throughput_tokens_per_s"]
    summary.update(
        {
            "real_active_output_throughput_tokens_per_s": real_throughput,
            "hisim_active_output_throughput_tokens_per_s": hisim_throughput,
            "active_throughput_error_percent": relative_error_percent(
                hisim_throughput, real_throughput
            ),
            "kv_cache_hit_rate_absolute_error": abs(
                hisim["summary"]["kv_cache_hit_rate"]
                - real["summary"]["kv_cache_hit_rate"]
            ),
        }
    )
    output = {
        "schema_version": 1,
        "evaluation_kind": "full_vllm_engine_hook_held_out",
        "real_source": str(args.real),
        "hisim_source": str(args.hisim),
        "summary": summary,
        "requests": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
