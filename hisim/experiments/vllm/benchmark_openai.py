#!/usr/bin/env python3
"""Measure a local vLLM OpenAI server with an exact token workload.

The benchmark sends a cold request followed by identical warm requests.  It
records client-side streaming timestamps and deltas of vLLM's server-side
prefix-cache counters.  The emitted prompt token IDs are also the workload for
the scheduler simulator, avoiding tokenizer drift during calibration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import aiohttp
import torch
import transformers
import vllm
from transformers import AutoTokenizer


QUERY_METRIC = "vllm:prefix_cache_queries_total"
HIT_METRIC = "vllm:prefix_cache_hits_total"
TTFT_SUM_METRIC = "vllm:time_to_first_token_seconds_sum"
TTFT_COUNT_METRIC = "vllm:time_to_first_token_seconds_count"
TPOT_SUM_METRIC = "vllm:request_time_per_output_token_seconds_sum"
TPOT_COUNT_METRIC = "vllm:request_time_per_output_token_seconds_count"
ITL_SUM_METRIC = "vllm:inter_token_latency_seconds_sum"
ITL_COUNT_METRIC = "vllm:inter_token_latency_seconds_count"
E2E_SUM_METRIC = "vllm:e2e_request_latency_seconds_sum"
E2E_COUNT_METRIC = "vllm:e2e_request_latency_seconds_count"


@dataclass
class RequestMeasurement:
    request_id: str
    start_offset_ms: float
    prompt_token_ids: list[int]
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float | None
    e2e_latency_ms: float
    client_ttft_ms: float
    client_tpot_ms: float | None
    client_e2e_latency_ms: float
    server_inter_token_latency_ms: float | None
    token_chunk_timestamps_ms: list[float]
    prefix_cache_queries: int
    prefix_cache_hits: int
    kv_cache_hit_rate: float
    finish_reason: str | None


def parse_metric(text: str, metric_name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        sample_name = line.split("{", 1)[0].split(" ", 1)[0]
        if sample_name != metric_name:
            continue
        total += float(line.rsplit(" ", 1)[-1])
    return total


def scrape_metrics(base_url: str) -> dict[str, float]:
    with urlopen(f"{base_url}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
    return {
        "prefix_cache_queries": parse_metric(text, QUERY_METRIC),
        "prefix_cache_hits": parse_metric(text, HIT_METRIC),
        "ttft_sum_seconds": parse_metric(text, TTFT_SUM_METRIC),
        "ttft_count": parse_metric(text, TTFT_COUNT_METRIC),
        "tpot_sum_seconds": parse_metric(text, TPOT_SUM_METRIC),
        "tpot_count": parse_metric(text, TPOT_COUNT_METRIC),
        "itl_sum_seconds": parse_metric(text, ITL_SUM_METRIC),
        "itl_count": parse_metric(text, ITL_COUNT_METRIC),
        "e2e_sum_seconds": parse_metric(text, E2E_SUM_METRIC),
        "e2e_count": parse_metric(text, E2E_COUNT_METRIC),
    }


def histogram_delta_ms(
    before: dict[str, float],
    after: dict[str, float],
    sum_key: str,
    count_key: str,
) -> float | None:
    count = after[count_key] - before[count_key]
    if count <= 0:
        return None
    return (after[sum_key] - before[sum_key]) / count * 1000.0


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def make_prompt_token_ids(
    tokenizer: Any, prefix_tokens: int, suffix_tokens: int, prompt_tag: str
) -> list[int]:
    seed_text = prompt_tag + ": " + (
        "KV cache stores reusable attention keys and values for efficient "
        "large language model inference. "
    )
    token_ids = tokenizer.encode(seed_text * (prefix_tokens + suffix_tokens), add_special_tokens=False)
    required = prefix_tokens + suffix_tokens
    if len(token_ids) < required:
        raise RuntimeError("failed to construct the requested prompt length")
    return token_ids[:required]


async def send_completion(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    request_id: str,
    prompt_token_ids: list[int],
    output_tokens: int,
    benchmark_start: float,
) -> tuple[RequestMeasurement, dict[str, float]]:
    metrics_before = scrape_metrics(base_url)
    request_start = time.perf_counter()
    payload = {
        "model": model,
        "request_id": request_id,
        "prompt": prompt_token_ids,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    token_times = []
    usage: dict[str, Any] = {}
    finish_reason = None
    async with session.post(
        f"{base_url}/v1/completions", json=payload, timeout=aiohttp.ClientTimeout(total=120)
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"request {request_id} failed ({response.status}): {body}")
        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                if choice.get("text"):
                    token_times.append(time.perf_counter())
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

    request_end = time.perf_counter()
    # The Prometheus logger is updated in the output path.  Yield once so the
    # metrics HTTP handler sees the completed scheduler stats.
    await asyncio.sleep(0.05)
    metrics_after = scrape_metrics(base_url)
    query_delta = round(
        metrics_after["prefix_cache_queries"] - metrics_before["prefix_cache_queries"]
    )
    hit_delta = round(
        metrics_after["prefix_cache_hits"] - metrics_before["prefix_cache_hits"]
    )
    completion_tokens = int(usage.get("completion_tokens", output_tokens))
    prompt_tokens = int(usage.get("prompt_tokens", len(prompt_token_ids)))
    if not token_times:
        raise RuntimeError(f"request {request_id} returned no streamed text chunks")
    client_ttft_ms = (token_times[0] - request_start) * 1000.0
    client_tpot_ms = None
    if completion_tokens > 1:
        client_tpot_ms = (token_times[-1] - token_times[0]) * 1000.0 / (
            completion_tokens - 1
        )
    client_e2e_latency_ms = (request_end - request_start) * 1000.0
    server_ttft_ms = histogram_delta_ms(
        metrics_before, metrics_after, "ttft_sum_seconds", "ttft_count"
    )
    server_tpot_ms = histogram_delta_ms(
        metrics_before, metrics_after, "tpot_sum_seconds", "tpot_count"
    )
    server_itl_ms = histogram_delta_ms(
        metrics_before, metrics_after, "itl_sum_seconds", "itl_count"
    )
    server_e2e_ms = histogram_delta_ms(
        metrics_before, metrics_after, "e2e_sum_seconds", "e2e_count"
    )
    measurement = RequestMeasurement(
        request_id=request_id,
        start_offset_ms=(request_start - benchmark_start) * 1000.0,
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        ttft_ms=server_ttft_ms if server_ttft_ms is not None else client_ttft_ms,
        tpot_ms=server_tpot_ms if server_tpot_ms is not None else client_tpot_ms,
        e2e_latency_ms=(
            server_e2e_ms if server_e2e_ms is not None else client_e2e_latency_ms
        ),
        client_ttft_ms=client_ttft_ms,
        client_tpot_ms=client_tpot_ms,
        client_e2e_latency_ms=client_e2e_latency_ms,
        server_inter_token_latency_ms=server_itl_ms,
        token_chunk_timestamps_ms=[
            (timestamp - request_start) * 1000.0 for timestamp in token_times
        ],
        prefix_cache_queries=query_delta,
        prefix_cache_hits=hit_delta,
        kv_cache_hit_rate=hit_delta / query_delta if query_delta else 0.0,
        finish_reason=finish_reason,
    )
    return measurement, metrics_after


def nvidia_smi_snapshot() -> str | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    benchmark_start = time.perf_counter()
    initial_metrics = scrape_metrics(args.base_url)
    measurements = []
    async with aiohttp.ClientSession() as session:
        for group in range(args.groups):
            prompt_token_ids = make_prompt_token_ids(
                tokenizer,
                args.prefix_tokens,
                args.suffix_tokens,
                f"{args.prompt_tag}-group-{group}",
            )
            for index in range(args.warm_repeats + 1):
                phase = "cold" if index == 0 else f"warm-{index}"
                request_id = f"group-{group}-{phase}"
                measurement, _ = await send_completion(
                    session,
                    base_url=args.base_url,
                    model=args.model,
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    output_tokens=args.output_tokens,
                    benchmark_start=benchmark_start,
                )
                measurements.append(measurement)
    final_metrics = scrape_metrics(args.base_url)

    ttfts = [measurement.ttft_ms for measurement in measurements]
    tpots = [
        measurement.tpot_ms
        for measurement in measurements
        if measurement.tpot_ms is not None
    ]
    total_output_tokens = sum(measurement.output_tokens for measurement in measurements)
    active_request_time_ms = sum(
        measurement.e2e_latency_ms for measurement in measurements
    )
    request_window_ms = (
        measurements[-1].start_offset_ms
        + measurements[-1].e2e_latency_ms
        - measurements[0].start_offset_ms
    )
    prefix_queries = round(
        final_metrics["prefix_cache_queries"] - initial_metrics["prefix_cache_queries"]
    )
    prefix_hits = round(
        final_metrics["prefix_cache_hits"] - initial_metrics["prefix_cache_hits"]
    )
    return {
        "schema_version": 1,
        "backend": "vllm-openai-real",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_smi": nvidia_smi_snapshot(),
        },
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "prompt_tokens": args.prefix_tokens + args.suffix_tokens,
            "output_tokens": args.output_tokens,
            "warm_repeats": args.warm_repeats,
            "prompt_tag": args.prompt_tag,
            "groups": args.groups,
            "sequential": True,
            "timing_source": "vllm_prometheus_histogram_delta",
        },
        "summary": {
            "request_count": len(measurements),
            "output_tokens": total_output_tokens,
            "request_window_ms": request_window_ms,
            "observed_output_throughput_tokens_per_s": (
                total_output_tokens / (request_window_ms / 1000.0)
            ),
            "active_request_time_ms": active_request_time_ms,
            "active_output_throughput_tokens_per_s": (
                total_output_tokens / (active_request_time_ms / 1000.0)
            ),
            "ttft_mean_ms": statistics.fmean(ttfts),
            "ttft_p50_ms": percentile(ttfts, 0.50),
            "ttft_p90_ms": percentile(ttfts, 0.90),
            "tpot_mean_ms": statistics.fmean(tpots) if tpots else None,
            "tpot_p50_ms": percentile(tpots, 0.50),
            "tpot_p90_ms": percentile(tpots, 0.90),
            "prefix_cache_queries": prefix_queries,
            "prefix_cache_hits": prefix_hits,
            "kv_cache_hit_rate": prefix_hits / prefix_queries if prefix_queries else 0.0,
        },
        "requests": [asdict(measurement) for measurement in measurements],
        "metrics_before": initial_metrics,
        "metrics_after": final_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warm-repeats", type=int, default=4)
    parser.add_argument("--prompt-tag", default="calibration-v1")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
