#!/usr/bin/env python3
"""Fit long-context recomputation and tier I/O costs from real measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_nonnegative_quadratic(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    design = np.column_stack([xs, xs**2])
    linear, quadratic = np.linalg.lstsq(design, ys, rcond=None)[0]
    if linear < 0:
        linear = 0.0
        quadratic = max(0.0, float(np.dot(xs**2, ys) / np.dot(xs**2, xs**2)))
    elif quadratic < 0:
        quadratic = 0.0
        linear = max(0.0, float(np.dot(xs, ys) / np.dot(xs, xs)))
    return float(linear), float(quadratic)


def split_phase(request_id: str) -> tuple[str, str]:
    if request_id.endswith("-cold"):
        return request_id[: -len("-cold")], "cold"
    marker = "-warm-"
    if marker in request_id:
        return request_id.split(marker, 1)[0], "warm"
    raise ValueError(f"unknown request phase: {request_id}")


def collect_length_point(document: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for request in document["requests"]:
        group, phase = split_phase(request["request_id"])
        groups.setdefault(group, {})[phase] = request
    samples = []
    for group, requests in groups.items():
        if set(requests) != {"cold", "warm"}:
            raise ValueError(f"cold/warm pair incomplete: {group}")
        cold = requests["cold"]
        warm = requests["warm"]
        hit_tokens = warm["prefix_cache_hits"]
        if hit_tokens <= 0:
            raise ValueError(f"warm request has no prefix hit: {group}")
        samples.append(
            {
                "group": group,
                "hit_tokens": hit_tokens,
                "cold_ttft_ms": cold["ttft_ms"],
                "warm_ttft_ms": warm["ttft_ms"],
                "saved_compute_ms": cold["ttft_ms"] - warm["ttft_ms"],
            }
        )
    saved = [sample["saved_compute_ms"] for sample in samples]
    hit_tokens = [sample["hit_tokens"] for sample in samples]
    if len(set(hit_tokens)) != 1:
        raise ValueError("warm hit tokens differ within one length")
    return {
        "prompt_tokens": document["config"]["prompt_tokens"],
        "hit_tokens": hit_tokens[0],
        "sample_count": len(samples),
        "saved_compute_ms_samples": saved,
        "saved_compute_ms_median": max(0.0, statistics.median(saved)),
        "saved_compute_ms_mean": statistics.fmean(saved),
        "cold_ttft_ms_mean": statistics.fmean(
            sample["cold_ttft_ms"] for sample in samples
        ),
        "warm_ttft_ms_mean": statistics.fmean(
            sample["warm_ttft_ms"] for sample in samples
        ),
    }


def evaluate(points: list[dict[str, Any]], linear: float, quadratic: float) -> dict:
    errors = []
    records = []
    for point in points:
        x = point["hit_tokens"]
        actual = point["saved_compute_ms_median"]
        predicted = linear * x + quadratic * x**2
        error = predicted - actual
        errors.append(error)
        records.append(
            {
                "prompt_tokens": point["prompt_tokens"],
                "hit_tokens": x,
                "actual_saved_compute_ms": actual,
                "predicted_saved_compute_ms": predicted,
                "absolute_error_ms": abs(error),
            }
        )
    return {
        "mae_ms": statistics.fmean(abs(error) for error in errors),
        "rmse_ms": math.sqrt(statistics.fmean(error**2 for error in errors)),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, action="append", type=Path)
    parser.add_argument("--io", required=True, type=Path)
    parser.add_argument("--config-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--dram-capacity-gib", type=float, default=64.0)
    parser.add_argument("--ssd-capacity-gib", type=float, default=512.0)
    args = parser.parse_args()

    documents = [load_json(path) for path in args.real]
    points = sorted(
        [collect_length_point(document) for document in documents],
        key=lambda point: point["prompt_tokens"],
    )
    fit_points = [
        (point["hit_tokens"], point["saved_compute_ms_median"])
        for point in points
    ]
    linear, quadratic = fit_nonnegative_quadratic(fit_points)
    in_sample = evaluate(points, linear, quadratic)

    leave_one_out = []
    if len(points) >= 3:
        for index, held_out in enumerate(points):
            training = [point for i, point in enumerate(fit_points) if i != index]
            fold_linear, fold_quadratic = fit_nonnegative_quadratic(training)
            prediction = (
                fold_linear * held_out["hit_tokens"]
                + fold_quadratic * held_out["hit_tokens"] ** 2
            )
            actual = held_out["saved_compute_ms_median"]
            leave_one_out.append(
                {
                    "prompt_tokens": held_out["prompt_tokens"],
                    "actual_saved_compute_ms": actual,
                    "predicted_saved_compute_ms": prediction,
                    "absolute_error_ms": abs(prediction - actual),
                }
            )

    io = load_json(args.io)
    h2d = io["hbm_dram"]["h2d_fit"]
    d2h = io["hbm_dram"]["d2h_fit"]
    ssd = io["ssd"]
    derived = io["derived"]
    config = {
        "dram_capacity_bytes": int(args.dram_capacity_gib * 1024**3),
        "ssd_capacity_bytes": int(args.ssd_capacity_gib * 1024**3),
        "eviction_policy": "lru",
        "prefetch_policy": "cost_aware",
        "write_policy": "write_through",
        "recompute_ms_per_token": linear,
        "recompute_ms_per_token_squared": quadratic,
        "min_savings_ms": 1.0,
        "max_prefetch_blocks": 0,
        "dram_read_bandwidth_gbps": h2d["bandwidth_gbps"],
        "dram_write_bandwidth_gbps": d2h["bandwidth_gbps"],
        "ssd_read_bandwidth_gbps": derived[
            "ssd_to_hbm_serial_bandwidth_gbps"
        ],
        "ssd_write_bandwidth_gbps": ssd["write"]["bandwidth_gbps"],
        "dram_read_latency_ms": h2d["fixed_latency_ms"],
        "dram_write_latency_ms": d2h["fixed_latency_ms"],
        "ssd_read_latency_ms": h2d["fixed_latency_ms"],
        "ssd_write_latency_ms": 0.0,
        "prefetch_overlap_fraction": 0.0,
        "write_overlap_fraction": 1.0,
        "parallel_tier_reads": False,
        "ssd_state_path": "/tmp/hisim-vllm-ssd-state.json",
        "metrics_path": "/tmp/hisim-vllm-tiered-metrics.json",
    }
    report = {
        "schema_version": 1,
        "experiment": "rtx4090-long-context-tiered-kv-calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": documents[0]["environment"],
        "model": documents[0]["config"]["model"],
        "method": {
            "recompute_cost": (
                "median(cold TTFT - warm TTFT) = linear * hit_tokens + "
                "quadratic * hit_tokens^2"
            ),
            "io_cost": "pinned CUDA copies plus fio direct I/O at KV block size",
            "batch_size": 1,
        },
        "sources": {
            "real": [
                {"path": str(path), "sha256": sha256(path)} for path in args.real
            ],
            "io": {"path": str(args.io), "sha256": sha256(args.io)},
        },
        "length_points": points,
        "fit": {
            "recompute_ms_per_token": linear,
            "recompute_ms_per_token_squared": quadratic,
            "in_sample": in_sample,
            "leave_one_length_out": {
                "mae_ms": statistics.fmean(
                    record["absolute_error_ms"] for record in leave_one_out
                ),
                "records": leave_one_out,
            },
        },
        "io_summary": {
            "dram_to_hbm": h2d,
            "hbm_to_dram": d2h,
            "ssd_read": ssd["read"],
            "ssd_write": ssd["write"],
            **derived,
        },
        "generated_tiered_config": config,
        "limitations": [
            "Single-request sequential Qwen3-0.6B measurements.",
            "The SSD path is modeled as serial SSD->DRAM plus DRAM->HBM.",
            "I/O contention and overlap need separate concurrent calibration.",
        ],
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["fit"], ensure_ascii=False))


if __name__ == "__main__":
    main()
