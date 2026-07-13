#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

SOURCE_WORKLOAD=${SOURCE_WORKLOAD:-/home/songxy/workspace/KVSim-LLM/ref/simulator_sources/LLMServingSim-main/workloads/sharegpt-qwen3-32b-300-sps10.jsonl}
DATASET_PATH=${DATASET_PATH:-"$ROOT_DIR/test/assets/llmservingsim/sharegpt-qwen3-32b-300-sps10.hisim.jsonl"}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-32B-FP8}
SIM_CONFIG=${SIM_CONFIG:-"$ROOT_DIR/configs/simulation/qwen3_32b_h100_aic.json"}
NUM_PROMPTS=${NUM_PROMPTS:-300}
PYTHON_BIN=${PYTHON_BIN:-python}
HISIM_CONDA_ENV=${HISIM_CONDA_ENV:-hisim}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30001}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/test/results/llmservingsim_sharegpt_300_hicache/${RADIX_EVICTION_POLICY:-lru}"}
RUN_NAME=${RUN_NAME:-"sharegpt-qwen3-32b-300-sps10-${RADIX_EVICTION_POLICY:-lru}"}
START_SERVER=${HISIM_START_SERVER:-1}

ENABLE_HIERARCHICAL_CACHE=${ENABLE_HIERARCHICAL_CACHE:-1}
RADIX_EVICTION_POLICY=${RADIX_EVICTION_POLICY:-lru}
HICACHE_STORAGE_BACKEND=${HICACHE_STORAGE_BACKEND:-file}
HICACHE_STORAGE_PREFETCH_POLICY=${HICACHE_STORAGE_PREFETCH_POLICY:-best_effort}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-write_through}
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-kernel}
HICACHE_MEM_LAYOUT=${HICACHE_MEM_LAYOUT:-layer_first}
HICACHE_STORAGE_BACKEND_EXTRA_CONFIG=${HICACHE_STORAGE_BACKEND_EXTRA_CONFIG:-}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32768}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
HICACHE_RATIO=${HICACHE_RATIO:-}
HICACHE_SIZE_GB=${HICACHE_SIZE_GB:-}

PYTHON_CMD=("$PYTHON_BIN")
if ! "$PYTHON_BIN" -c "import hisim" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    CONDA_PYTHON="$CONDA_BASE/envs/$HISIM_CONDA_ENV/bin/python"
    if [[ -x "$CONDA_PYTHON" ]]; then
      PYTHON_CMD=("$CONDA_PYTHON")
    else
      PYTHON_CMD=(conda run -n "$HISIM_CONDA_ENV" --no-capture-output python)
    fi
  fi
fi

mkdir -p "$(dirname "$DATASET_PATH")" "$OUT_DIR"

if [[ ! -s "$DATASET_PATH" || "${RECONVERT_WORKLOAD:-0}" == "1" ]]; then
  "${PYTHON_CMD[@]}" "$ROOT_DIR/test/scripts/llmservingsim_to_hisim.py" \
    -i "$SOURCE_WORKLOAD" \
    -o "$DATASET_PATH" \
    --max-requests "$NUM_PROMPTS"
else
  echo "Using preconverted Hisim workload: $DATASET_PATH"
fi

export SGLANG_USE_CPU_ENGINE=${SGLANG_USE_CPU_ENGINE:-1}
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
export HISIM_OUTPUT_DIR=${HISIM_OUTPUT_DIR:-"$OUT_DIR/hisim_output"}
export HISIM_SIMULATION_MODE=${HISIM_SIMULATION_MODE:-OFFLINE}
export HISIM_HICACHE_STORAGE_PATH=${HISIM_HICACHE_STORAGE_PATH:-"$OUT_DIR/hicache_storage_keys.txt"}
export HISIM_RESET_HICACHE_STORAGE=${HISIM_RESET_HICACHE_STORAGE:-1}

RESULT_PATH="$OUT_DIR/${RUN_NAME}.result.jsonl"
PRETTY_RESULT_PATH=${PRETTY_RESULT_PATH:-"$OUT_DIR/${RUN_NAME}.result.pretty.json"}
SUMMARY_PATH="$OUT_DIR/${RUN_NAME}.summary.json"
SERVER_LOG="$OUT_DIR/server.log"

rm -f "$RESULT_PATH" "$PRETTY_RESULT_PATH" "$SUMMARY_PATH"
rm -rf "$HISIM_OUTPUT_DIR"
mkdir -p "$HISIM_OUTPUT_DIR"

echo "============ Hisim ShareGPT HiCache Replay ============"
echo "Workload:                       $DATASET_PATH"
echo "Source workload:                $SOURCE_WORKLOAD"
echo "Requests:                       $NUM_PROMPTS"
echo "Simulation config:              $SIM_CONFIG"
echo "Output dir:                     $OUT_DIR"
echo "HiSim output dir:               $HISIM_OUTPUT_DIR"
echo "HiCache storage file:           $HISIM_HICACHE_STORAGE_PATH"
echo "Reset storage before run:       $HISIM_RESET_HICACHE_STORAGE"
echo "Hierarchical cache enabled:     $ENABLE_HIERARCHICAL_CACHE"
echo "HiCache storage backend:        $HICACHE_STORAGE_BACKEND"
echo "HiCache storage prefetch:       $HICACHE_STORAGE_PREFETCH_POLICY"
echo "HiCache write policy:           $HICACHE_WRITE_POLICY"
echo "Radix eviction policy:          $RADIX_EVICTION_POLICY"
echo "Max total tokens:               $MAX_TOTAL_TOKENS"
echo "========================================================"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

if [[ "$START_SERVER" == "1" ]]; then
  SERVER_CMD=(
    "${PYTHON_CMD[@]}" -m hisim.simulation.sglang.launch_server
    --model-path "$MODEL_PATH"
    --sim-config-path "$SIM_CONFIG"
    --skip-server-warmup
    --device cpu
    --host "$HOST"
    --port "$PORT"
    --radix-eviction-policy "$RADIX_EVICTION_POLICY"
    --max-total-tokens "$MAX_TOTAL_TOKENS"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  )

  if [[ "$ENABLE_HIERARCHICAL_CACHE" == "1" ]]; then
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-storage-backend "$HICACHE_STORAGE_BACKEND"
      --hicache-storage-prefetch-policy "$HICACHE_STORAGE_PREFETCH_POLICY"
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
      --hicache-io-backend "$HICACHE_IO_BACKEND"
      --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
  fi
  if [[ -n "$HICACHE_STORAGE_BACKEND_EXTRA_CONFIG" ]]; then
    SERVER_CMD+=(--hicache-storage-backend-extra-config "$HICACHE_STORAGE_BACKEND_EXTRA_CONFIG")
  fi
  if [[ -n "$HICACHE_RATIO" ]]; then
    SERVER_CMD+=(--hicache-ratio "$HICACHE_RATIO")
  fi
  if [[ -n "$HICACHE_SIZE_GB" ]]; then
    SERVER_CMD+=(--hicache-size "$HICACHE_SIZE_GB")
  fi

  "${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  trap cleanup EXIT

  ready=0
  for _ in $(seq 1 120); do
    if curl -s --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "Hisim server exited before becoming ready. See $SERVER_LOG" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "Hisim server did not become ready. See $SERVER_LOG" >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
fi

"${PYTHON_CMD[@]}" -m hisim.simulation.bench_serving \
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

METRICS_PATH="$HISIM_OUTPUT_DIR/metrics.json"
REQUEST_STATS_PATH="$HISIM_OUTPUT_DIR/request.jsonl"
ITERATION_STATS_PATH="$HISIM_OUTPUT_DIR/iteration.jsonl"

"${PYTHON_CMD[@]}" - "$RESULT_PATH" "$PRETTY_RESULT_PATH" "$METRICS_PATH" "$SUMMARY_PATH" "$SERVER_LOG" "$HISIM_HICACHE_STORAGE_PATH" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
pretty_result_path = Path(sys.argv[2])
metrics_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
server_log = Path(sys.argv[5])
storage_path = Path(sys.argv[6])

last = json.loads(result_path.read_text(encoding="utf-8").strip().splitlines()[-1])
metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

server_info = last.get("server_info") or {}
storage_key_count = 0
if storage_path.exists():
    storage_key_count = sum(1 for line in storage_path.read_text(encoding="utf-8").splitlines() if line.strip())

log_text = server_log.read_text(encoding="utf-8", errors="ignore") if server_log.exists() else ""
summary = {
    "completed": last.get("completed"),
    "disk_prefetch_ratio": last.get("disk_prefetch_ratio"),
    "duration_s": last.get("duration"),
    "hicache_storage_file": str(storage_path),
    "hicache_storage_key_count": storage_key_count,
    "input_token_throughput": last.get("input_throughput"),
    "mean_e2e_latency_ms": last.get("mean_e2e_latency_ms"),
    "mean_itl_ms": last.get("mean_itl_ms"),
    "mean_tpot_ms": last.get("mean_tpot_ms"),
    "mean_ttft_ms": last.get("mean_ttft_ms"),
    "oom_estimator_warnings": log_text.count("Out of memory detected during estimation"),
    "output_token_throughput": last.get("output_throughput"),
    "p99_e2e_latency_ms": last.get("p99_e2e_latency_ms"),
    "p99_itl_ms": last.get("p99_itl_ms"),
    "p99_tpot_ms": last.get("p99_tpot_ms"),
    "p99_ttft_ms": last.get("p99_ttft_ms"),
    "prefix_cache_reused_ratio": last.get("prefix_cache_reused_ratio"),
    "radix_eviction_policy": server_info.get("radix_eviction_policy"),
    "request_throughput": last.get("request_throughput"),
    "server_args": {
        key: server_info.get(key)
        for key in [
            "enable_hierarchical_cache",
            "hicache_storage_backend",
            "hicache_storage_prefetch_policy",
            "hicache_write_policy",
            "hicache_io_backend",
            "hicache_mem_layout",
            "radix_eviction_policy",
            "max_total_tokens",
            "max_total_num_tokens",
            "max_prefill_tokens",
            "chunked_prefill_size",
            "device",
        ]
    },
    "total_input_tokens": last.get("total_input_tokens"),
    "total_output_tokens": last.get("total_output_tokens"),
}
if metrics:
    summary["server_metrics"] = {
        key: metrics.get(key)
        for key in [
            "num_requests",
            "completed",
            "duration",
            "prefix_cache_reused_ratio",
            "disk_prefetch_ratio",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_e2e_latency_ms",
            "time_cost",
        ]
    }

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def fmt_ratio(value):
    if value is None:
        return "unavailable"
    return f"{value:.6f} ({value * 100:.2f}%)"

def fmt_float(value, suffix=""):
    if value is None:
        return "unavailable"
    return f"{value:.2f}{suffix}"

print("\n============ Hisim ShareGPT HiCache Summary ============")
print(f"Completed requests:                 {summary['completed']}")
print(f"Total input tokens:                  {summary['total_input_tokens']}")
print(f"Total output tokens:                 {summary['total_output_tokens']}")
print(f"Prefix cache reused ratio:           {fmt_ratio(summary['prefix_cache_reused_ratio'])}")
print(f"Disk prefetch ratio:                 {fmt_ratio(summary['disk_prefetch_ratio'])}")
print(f"Mean TTFT:                           {fmt_float(summary['mean_ttft_ms'], ' ms')}")
print(f"Mean TPOT:                           {fmt_float(summary['mean_tpot_ms'], ' ms')}")
print(f"Mean E2E latency:                    {fmt_float(summary['mean_e2e_latency_ms'], ' ms')}")
print(f"Storage keys after run:              {summary['hicache_storage_key_count']}")
print(f"OOM estimator warnings:              {summary['oom_estimator_warnings']}")
print(f"Result JSONL:                        {result_path}")
print(f"Pretty result JSON:                  {pretty_result_path}")
print(f"Server metrics:                      {metrics_path}")
print(f"Run summary:                         {summary_path}")
print("=========================================================")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
