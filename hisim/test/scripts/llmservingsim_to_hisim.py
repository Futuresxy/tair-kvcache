#!/usr/bin/env python3
"""Convert LLMServingSim JSONL workloads to Hisim collection JSONL.

LLMServingSim flat records:
  {"input_toks": int, "output_toks": int, "arrival_time_ns": int,
   "input_tok_ids": [...], "output_tok_ids": [...]}

LLMServingSim agentic session records:
  {"session_id": str, "arrival_time_ns": int, "sub_requests": [...]}

Hisim collection records:
  {"rid": str, "timestamp": float, "input_length": int, "output_length": int,
   "input_ids": [...], "output_ids": [...], "queue_end": float,
   "final_prefix_cache_len": int}
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NS_PER_SEC = 1_000_000_000


@dataclass
class ConvertStats:
    converted: int = 0
    skipped_missing_ids: int = 0
    synthetic_ids: int = 0
    flat_records: int = 0
    sessions: int = 0
    sub_requests: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LLMServingSim workloads to Hisim collection format."
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSONL workload.")
    parser.add_argument("-o", "--output", required=True, help="Output Hisim JSONL.")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Stop after this many converted requests. 0 means no limit.",
    )
    parser.add_argument(
        "--min-input-length",
        type=int,
        default=0,
        help="Skip requests whose input length is below this value.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=0,
        help="Skip requests whose input length is above this value. 0 disables.",
    )
    parser.add_argument(
        "--agentic-gap-sec",
        type=float,
        default=1.0,
        help=(
            "Additional timestamp gap inserted between sub_requests in one "
            "agentic session. Hisim replay has no dependency graph, so a "
            "positive gap preserves request order more realistically."
        ),
    )
    parser.add_argument(
        "--ignore-tool-duration",
        action="store_true",
        help=(
            "Do not add sub_request.tool_duration_ns to following "
            "sub-request timestamps."
        ),
    )
    parser.add_argument(
        "--preserve-timestamps",
        action="store_true",
        help=(
            "Keep absolute converted timestamps. By default timestamps are "
            "normalized to start at 0."
        ),
    )
    parser.add_argument(
        "--synthetic-missing-token-ids",
        action="store_true",
        help=(
            "Synthesize deterministic token ids when input_tok_ids is missing. "
            "This keeps lengths but does not preserve real prefix-cache behavior."
        ),
    )
    parser.add_argument(
        "--synthetic-vocab-size",
        type=int,
        default=32000,
        help="Vocabulary range used with --synthetic-missing-token-ids.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def synthetic_ids(length: int, request_index: int, vocab_size: int) -> list[int]:
    high = max(vocab_size - 1000, 1)
    return [
        1000 + ((request_index * 1_315_423_911 + pos) % high)
        for pos in range(length)
    ]


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return n


def best_prior_prefix_lens(records: list[dict[str, Any]]) -> list[int]:
    root: dict[int, Any] = {}
    lens: list[int] = []

    for record in records:
        node = root
        best = 0
        for token in record["input_ids"]:
            next_node = node.get(token)
            if next_node is None:
                break
            node = next_node
            best += 1
        lens.append(best)

        node = root
        for token in record["input_ids"]:
            node = node.setdefault(token, {})

    return lens


def make_hisim_record(
    *,
    rid: str,
    timestamp_s: float,
    source: dict[str, Any],
    stats: ConvertStats,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    input_ids = source.get("input_tok_ids")
    input_len = int(source.get("input_toks", len(input_ids) if input_ids else 0))

    if input_ids is None:
        if not args.synthetic_missing_token_ids:
            stats.skipped_missing_ids += 1
            return None
        input_ids = synthetic_ids(
            input_len, stats.converted, args.synthetic_vocab_size
        )
        stats.synthetic_ids += 1
    else:
        input_ids = list(input_ids)
        input_len = len(input_ids)

    if input_len < args.min_input_length:
        return None
    if args.max_input_length and input_len > args.max_input_length:
        return None

    output_ids = list(source.get("output_tok_ids") or [])
    output_len = int(source.get("output_toks", len(output_ids)))

    return {
        "rid": rid,
        "timestamp": timestamp_s,
        "input_length": input_len,
        "output_length": output_len,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "queue_end": timestamp_s,
        "final_prefix_cache_len": 0,
    }


def convert(args: argparse.Namespace) -> tuple[list[dict[str, Any]], ConvertStats]:
    stats = ConvertStats()
    records: list[dict[str, Any]] = []

    def reached_limit() -> bool:
        return bool(args.max_requests and stats.converted >= args.max_requests)

    for row_idx, row in enumerate(load_jsonl(Path(args.input))):
        if reached_limit():
            break

        if "sub_requests" not in row:
            stats.flat_records += 1
            record = make_hisim_record(
                rid=f"flat-{row_idx}",
                timestamp_s=float(row.get("arrival_time_ns", 0)) / NS_PER_SEC,
                source=row,
                stats=stats,
                args=args,
            )
            if record is not None:
                records.append(record)
                stats.converted += 1
        else:
            stats.sessions += 1
            session_id = row.get("session_id", f"session-{row_idx}")
            timestamp_s = float(row.get("arrival_time_ns", 0)) / NS_PER_SEC
            for sub_idx, sub_request in enumerate(row.get("sub_requests") or []):
                if reached_limit():
                    break

                stats.sub_requests += 1
                record = make_hisim_record(
                    rid=f"{session_id}-sub{sub_idx}",
                    timestamp_s=timestamp_s,
                    source=sub_request,
                    stats=stats,
                    args=args,
                )
                if record is not None:
                    records.append(record)
                    stats.converted += 1

                tool_duration_s = 0.0
                if not args.ignore_tool_duration:
                    tool_duration_s = (
                        float(sub_request.get("tool_duration_ns", 0)) / NS_PER_SEC
                    )
                timestamp_s += max(args.agentic_gap_sec, 0.0) + tool_duration_s

    records.sort(key=lambda item: item["timestamp"])
    if records and not args.preserve_timestamps:
        min_timestamp = records[0]["timestamp"]
        for record in records:
            record["timestamp"] -= min_timestamp
            record["queue_end"] = record["timestamp"]

    return records, stats


def print_summary(records: list[dict[str, Any]], stats: ConvertStats) -> None:
    input_lens = [record["input_length"] for record in records]
    output_lens = [record["output_length"] for record in records]
    adjacent_prefix = [
        common_prefix_len(left["input_ids"], right["input_ids"])
        for left, right in zip(records, records[1:])
    ]
    best_prior_prefix = best_prior_prefix_lens(records)

    def pct(values: list[int], p: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p / 100))]

    print(
        "Converted "
        f"{stats.converted} requests "
        f"(flat={stats.flat_records}, sessions={stats.sessions}, sub_requests={stats.sub_requests})."
    )
    if stats.skipped_missing_ids:
        print(f"Skipped {stats.skipped_missing_ids} requests without input_tok_ids.")
    if stats.synthetic_ids:
        print(
            f"Synthesized token ids for {stats.synthetic_ids} requests; "
            "these do not preserve real prefix-cache behavior."
        )
    if records:
        print(
            "Input tokens: "
            f"avg={statistics.mean(input_lens):.1f}, "
            f"p50={pct(input_lens, 50)}, p90={pct(input_lens, 90)}, max={max(input_lens)}"
        )
        print(
            "Output tokens: "
            f"avg={statistics.mean(output_lens):.1f}, "
            f"p50={pct(output_lens, 50)}, p90={pct(output_lens, 90)}, max={max(output_lens)}"
        )
        if adjacent_prefix:
            print(
                "Adjacent common prefix: "
                f"avg={statistics.mean(adjacent_prefix):.1f}, "
                f"p50={pct(adjacent_prefix, 50)}, "
                f"p90={pct(adjacent_prefix, 90)}, max={max(adjacent_prefix)}"
            )
        if best_prior_prefix:
            print(
                "Best prior common prefix: "
                f"avg={statistics.mean(best_prior_prefix):.1f}, "
                f"p50={pct(best_prior_prefix, 50)}, "
                f"p90={pct(best_prior_prefix, 90)}, max={max(best_prior_prefix)}"
            )


def main() -> int:
    args = parse_args()
    records, stats = convert(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}")
    print_summary(records, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
