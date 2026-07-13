#!/usr/bin/env python3
"""Calibrate HBM/DRAM/SSD transfer costs for HiSim tiered KV cache."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit_transfer(samples: list[dict[str, float]]) -> dict[str, float]:
    sizes = np.asarray([sample["bytes"] for sample in samples], dtype=np.float64)
    times = np.asarray(
        [sample["median_latency_ms"] for sample in samples], dtype=np.float64
    )
    design = np.column_stack([np.ones_like(sizes), sizes])
    intercept_ms, ms_per_byte = np.linalg.lstsq(design, times, rcond=None)[0]
    intercept_ms = max(0.0, float(intercept_ms))
    ms_per_byte = max(0.0, float(ms_per_byte))
    bandwidth_gbps = 1.0 / (ms_per_byte * 1_000_000.0) if ms_per_byte else 0.0
    return {
        "fixed_latency_ms": intercept_ms,
        "bandwidth_gbps": bandwidth_gbps,
    }


def measure_cuda_copy(
    *, sizes_mib: list[int], repeats: int, direction: str
) -> tuple[list[dict[str, float]], dict[str, float]]:
    samples = []
    for size_mib in sizes_mib:
        size_bytes = size_mib * 1024 * 1024
        host = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        if direction == "h2d":
            copy = lambda: device.copy_(host, non_blocking=True)
        else:
            copy = lambda: host.copy_(device, non_blocking=True)
        for _ in range(2):
            copy()
            torch.cuda.synchronize()
        latencies_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            copy()
            end.record()
            end.synchronize()
            latencies_ms.append(start.elapsed_time(end))
        median_latency_ms = statistics.median(latencies_ms)
        samples.append(
            {
                "size_mib": size_mib,
                "bytes": size_bytes,
                "latencies_ms": latencies_ms,
                "median_latency_ms": median_latency_ms,
                "median_bandwidth_gbps": size_bytes
                / median_latency_ms
                / 1_000_000.0,
            }
        )
        del host, device
    return samples, fit_transfer(samples)


def fio_operation(
    *, filename: Path, size_gib: int, block_kib: int, runtime_seconds: int, rw: str
) -> tuple[dict[str, Any], list[str]]:
    command = [
        "fio",
        f"--name=hisim-tier-{rw}",
        f"--filename={filename}",
        f"--size={size_gib}G",
        f"--bs={block_kib}k",
        f"--rw={rw}",
        "--direct=1",
        "--ioengine=libaio",
        "--iodepth=1",
        "--numjobs=1",
        f"--runtime={runtime_seconds}",
        "--time_based=1",
        "--group_reporting=1",
        "--output-format=json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stats = payload["jobs"][0]["read" if rw == "read" else "write"]
    latency = stats.get("clat_ns") or stats.get("lat_ns") or {}
    return (
        {
            "bandwidth_gbps": stats["bw_bytes"] / 1_000_000_000.0,
            "iops": stats["iops"],
            "io_bytes": stats["io_bytes"],
            "runtime_ms": stats["runtime"],
            "mean_completion_latency_ms": latency.get("mean", 0.0) / 1_000_000.0,
            "p50_completion_latency_ms": latency.get("percentile", {}).get(
                "50.000000", 0.0
            )
            / 1_000_000.0,
            "p99_completion_latency_ms": latency.get("percentile", {}).get(
                "99.000000", 0.0
            )
            / 1_000_000.0,
        },
        command,
    )


def serial_bandwidth_gbps(first: float, second: float) -> float:
    return 1.0 / (1.0 / first + 1.0 / second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ssd-directory", default="/tmp", type=Path)
    parser.add_argument("--ssd-size-gib", default=4, type=int)
    parser.add_argument("--ssd-runtime-seconds", default=5, type=int)
    parser.add_argument("--kv-block-kib", default=1792, type=int)
    parser.add_argument("--copy-sizes-mib", default="64,256,512")
    parser.add_argument("--copy-repeats", default=7, type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HBM/DRAM transfer calibration")
    sizes_mib = [int(value) for value in args.copy_sizes_mib.split(",")]

    h2d_samples, h2d_fit = measure_cuda_copy(
        sizes_mib=sizes_mib, repeats=args.copy_repeats, direction="h2d"
    )
    d2h_samples, d2h_fit = measure_cuda_copy(
        sizes_mib=sizes_mib, repeats=args.copy_repeats, direction="d2h"
    )
    args.ssd_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="hisim-tier-io-", dir=args.ssd_directory, delete=False
    ) as temporary:
        ssd_path = Path(temporary.name)
    try:
        ssd_write, write_command = fio_operation(
            filename=ssd_path,
            size_gib=args.ssd_size_gib,
            block_kib=args.kv_block_kib,
            runtime_seconds=args.ssd_runtime_seconds,
            rw="write",
        )
        ssd_read, read_command = fio_operation(
            filename=ssd_path,
            size_gib=args.ssd_size_gib,
            block_kib=args.kv_block_kib,
            runtime_seconds=args.ssd_runtime_seconds,
            rw="read",
        )
    finally:
        ssd_path.unlink(missing_ok=True)

    result = {
        "schema_version": 1,
        "experiment": "hisim-tier-io-calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "ssd_directory": str(args.ssd_directory.resolve()),
            "fio": command_output(["fio", "--version"]),
            "numa_policy": command_output(["numactl", "--show"]),
            "ssd_mount": command_output(
                [
                    "findmnt",
                    "-T",
                    str(args.ssd_directory.resolve()),
                    "-o",
                    "SOURCE,FSTYPE,TARGET",
                ]
            ),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
        "config": {
            "copy_sizes_mib": sizes_mib,
            "copy_repeats": args.copy_repeats,
            "ssd_size_gib": args.ssd_size_gib,
            "ssd_runtime_seconds": args.ssd_runtime_seconds,
            "kv_block_kib": args.kv_block_kib,
        },
        "hbm_dram": {
            "h2d_samples": h2d_samples,
            "h2d_fit": h2d_fit,
            "d2h_samples": d2h_samples,
            "d2h_fit": d2h_fit,
        },
        "ssd": {
            "read": ssd_read,
            "write": ssd_write,
            "read_command": read_command,
            "write_command": write_command,
        },
        "derived": {
            "ssd_to_hbm_serial_bandwidth_gbps": serial_bandwidth_gbps(
                ssd_read["bandwidth_gbps"], h2d_fit["bandwidth_gbps"]
            ),
            "hbm_to_ssd_serial_bandwidth_gbps": serial_bandwidth_gbps(
                d2h_fit["bandwidth_gbps"], ssd_write["bandwidth_gbps"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["derived"], ensure_ascii=False))


if __name__ == "__main__":
    main()
