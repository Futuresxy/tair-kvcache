#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEFAULT_PRECONVERTED_DATASET="$ROOT_DIR/test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-40.hisim.jsonl"
SOURCE_WORKLOAD=${SOURCE_WORKLOAD:-}
PRECONVERTED_DATASET=${PRECONVERTED_DATASET:-}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-32B-FP8}
SIM_CONFIG=${SIM_CONFIG:-"$ROOT_DIR/configs/simulation/qwen3_32b_h100_aic.json"}
NUM_PROMPTS=${NUM_PROMPTS:-40}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/test/results/llmservingsim_swebench"}
START_SERVER=${HISIM_START_SERVER:-1}

mkdir -p "$OUT_DIR"

if [[ -z "$PRECONVERTED_DATASET" && "$NUM_PROMPTS" == "40" ]]; then
  PRECONVERTED_DATASET="$DEFAULT_PRECONVERTED_DATASET"
fi

DATASET_PATH=${PRECONVERTED_DATASET:-"$OUT_DIR/swe-bench-qwen3-30b-a3b-${NUM_PROMPTS}.hisim.jsonl"}
RESULT_PATH="$OUT_DIR/swe-bench-qwen3-30b-a3b-${NUM_PROMPTS}.result.jsonl"
PRETTY_RESULT_PATH=${PRETTY_RESULT_PATH:-"$OUT_DIR/swe-bench-qwen3-30b-a3b-${NUM_PROMPTS}.result.pretty.json"}
SERVER_LOG="$OUT_DIR/server.log"

if [[ -z "$PRECONVERTED_DATASET" ]]; then
  if [[ -z "$SOURCE_WORKLOAD" ]]; then
    echo "SOURCE_WORKLOAD is required when PRECONVERTED_DATASET is not set." >&2
    echo "For the checked-in 40-request smoke test, use the default NUM_PROMPTS=40." >&2
    exit 1
  fi
  python "$ROOT_DIR/test/scripts/llmservingsim_to_hisim.py" \
    -i "$SOURCE_WORKLOAD" \
    -o "$DATASET_PATH" \
    --max-requests "$NUM_PROMPTS"
else
  echo "Using preconverted Hisim workload: $DATASET_PATH"
fi

rm -f "$RESULT_PATH" "$PRETTY_RESULT_PATH"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

if [[ "$START_SERVER" == "1" ]]; then
  export SGLANG_USE_CPU_ENGINE=${SGLANG_USE_CPU_ENGINE:-1}
  export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
  export HISIM_OUTPUT_DIR=${HISIM_OUTPUT_DIR:-"$OUT_DIR/hisim_output"}

  python -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --sim-config-path "$SIM_CONFIG" \
    --skip-server-warmup \
    --device cpu \
    --host "$HOST" \
    --port "$PORT" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  trap cleanup EXIT

  ready=0
  for _ in $(seq 1 90); do
    if curl -s --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "Hisim server did not become ready. See $SERVER_LOG" >&2
    exit 1
  fi
fi

python -m hisim.simulation.bench_serving \
  --warmup-requests 0 \
  --bench-mode simulation \
  --backend sglang \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --dataset-name hisim-collection \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --tokenize-prompt \
  --output-file "$RESULT_PATH" \
  --pretty-output-file "$PRETTY_RESULT_PATH"

METRICS_PATH="${HISIM_OUTPUT_DIR:-"$OUT_DIR/hisim_output"}/metrics.json"
python - "$RESULT_PATH" "$METRICS_PATH" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
last = json.loads(result_path.read_text(encoding="utf-8").strip().splitlines()[-1])
metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
summary = {
    "completed": last.get("completed"),
    "total_input_tokens": last.get("total_input_tokens"),
    "total_output_tokens": last.get("total_output_tokens"),
    "prefix_cache_reused_ratio": last.get("prefix_cache_reused_ratio"),
    "disk_prefetch_ratio": last.get("disk_prefetch_ratio"),
    "mean_ttft_ms": last.get("mean_ttft_ms"),
    "mean_tpot_ms": last.get("mean_tpot_ms"),
}
cache_ratio = summary["prefix_cache_reused_ratio"]

print("\n============ Hisim KVCache Summary ============")
print(f"Completed requests:                 {summary['completed']}")
print(f"Total input tokens:                  {summary['total_input_tokens']}")
print(f"Total output tokens:                 {summary['total_output_tokens']}")
if cache_ratio is None:
    print("Prefix cache reused ratio:           unavailable")
else:
    print(
        f"Prefix cache reused ratio:           {cache_ratio:.6f} "
        f"({cache_ratio * 100:.2f}%)"
    )
print(f"Disk prefetch ratio:                 {summary['disk_prefetch_ratio']}")
print(f"Mean TTFT (ms):                      {summary['mean_ttft_ms']:.2f}")
print(f"Mean TPOT (ms):                      {summary['mean_tpot_ms']:.2f}")
if metrics:
    print(f"Server metrics:                      {metrics_path}")
    print(f"Per-request stats:                   {out_dir / 'hisim_output/request.jsonl'}")
    print(f"Per-iteration stats:                 {out_dir / 'hisim_output/iteration.jsonl'}")
print("================================================")

print(json.dumps(summary, indent=2, sort_keys=True))
PY
