#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SINGLE_RUN_SCRIPT="$ROOT_DIR/test/scripts/run_llmservingsim_sharegpt_300_hicache.sh"

SOURCE_WORKLOAD=${SOURCE_WORKLOAD:-/home/songxy/workspace/KVSim-LLM/ref/simulator_sources/LLMServingSim-main/workloads/sharegpt-qwen3-32b-300-sps10.jsonl}
DATASET_PATH=${DATASET_PATH:-"$ROOT_DIR/test/assets/llmservingsim/sharegpt-qwen3-32b-300-sps10.hisim.jsonl"}
BASE_OUT_DIR=${BASE_OUT_DIR:-"$ROOT_DIR/test/results/llmservingsim_sharegpt_300_hicache"}
RADIX_EVICTION_POLICIES=${RADIX_EVICTION_POLICIES:-"lru lfu"}
PASSES=${PASSES:-"cold warm"}
PORT=${PORT:-30001}
NUM_PROMPTS=${NUM_PROMPTS:-300}
PYTHON_BIN=${PYTHON_BIN:-python}
HISIM_CONDA_ENV=${HISIM_CONDA_ENV:-hisim}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-32B-FP8}
SIM_CONFIG=${SIM_CONFIG:-"$ROOT_DIR/configs/simulation/qwen3_32b_h100_aic.json"}
ENABLE_HIERARCHICAL_CACHE=${ENABLE_HIERARCHICAL_CACHE:-1}
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

mkdir -p "$BASE_OUT_DIR"

echo "============ Hisim ShareGPT HiCache Matrix ============"
echo "Policies:                       $RADIX_EVICTION_POLICIES"
echo "Passes:                         $PASSES"
echo "Base output dir:                $BASE_OUT_DIR"
echo "Storage backend:                $HICACHE_STORAGE_BACKEND"
echo "Max total tokens:               $MAX_TOTAL_TOKENS"
echo "======================================================="

for policy in $RADIX_EVICTION_POLICIES; do
  storage_path="$BASE_OUT_DIR/$policy/hicache_storage_keys.txt"
  mkdir -p "$(dirname "$storage_path")"

  for pass_name in $PASSES; do
    reset_storage=0
    if [[ "$pass_name" == "cold" ]]; then
      reset_storage=1
    fi

    out_dir="$BASE_OUT_DIR/$policy/$pass_name"
    run_name="sharegpt-qwen3-32b-300-sps10-${policy}-${pass_name}"

    echo
    echo "---- Running policy=$policy pass=$pass_name reset_storage=$reset_storage ----"
    SOURCE_WORKLOAD="$SOURCE_WORKLOAD" \
      DATASET_PATH="$DATASET_PATH" \
      PYTHON_BIN="$PYTHON_BIN" \
      HISIM_CONDA_ENV="$HISIM_CONDA_ENV" \
      MODEL_PATH="$MODEL_PATH" \
      SIM_CONFIG="$SIM_CONFIG" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      PORT="$PORT" \
      OUT_DIR="$out_dir" \
      RUN_NAME="$run_name" \
      ENABLE_HIERARCHICAL_CACHE="$ENABLE_HIERARCHICAL_CACHE" \
      RADIX_EVICTION_POLICY="$policy" \
      HICACHE_STORAGE_BACKEND="$HICACHE_STORAGE_BACKEND" \
      HICACHE_STORAGE_PREFETCH_POLICY="$HICACHE_STORAGE_PREFETCH_POLICY" \
      HICACHE_WRITE_POLICY="$HICACHE_WRITE_POLICY" \
      HICACHE_IO_BACKEND="$HICACHE_IO_BACKEND" \
      HICACHE_MEM_LAYOUT="$HICACHE_MEM_LAYOUT" \
      HICACHE_STORAGE_BACKEND_EXTRA_CONFIG="$HICACHE_STORAGE_BACKEND_EXTRA_CONFIG" \
      MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
      MAX_PREFILL_TOKENS="$MAX_PREFILL_TOKENS" \
      CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
      HICACHE_RATIO="$HICACHE_RATIO" \
      HICACHE_SIZE_GB="$HICACHE_SIZE_GB" \
      HISIM_OUTPUT_DIR="$out_dir/hisim_output" \
      HISIM_HICACHE_STORAGE_PATH="$storage_path" \
      HISIM_RESET_HICACHE_STORAGE="$reset_storage" \
      bash "$SINGLE_RUN_SCRIPT"
  done
done

"${PYTHON_CMD[@]}" - "$BASE_OUT_DIR" "$RADIX_EVICTION_POLICIES" "$PASSES" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
policy_order = sys.argv[2].split()
pass_order = sys.argv[3].split()
summaries = sorted(base.glob("*/*/*.summary.json"))
rows_by_key = {}
for path in summaries:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "policy": path.parents[1].name,
        "pass": path.parent.name,
        "completed": data.get("completed"),
        "prefix_cache_reused_ratio": data.get("prefix_cache_reused_ratio"),
        "disk_prefetch_ratio": data.get("disk_prefetch_ratio"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "mean_e2e_latency_ms": data.get("mean_e2e_latency_ms"),
        "storage_keys": data.get("hicache_storage_key_count"),
        "summary": str(path),
    }
    rows_by_key[(row["policy"], row["pass"])] = row

rows = []
for policy in policy_order:
    for pass_name in pass_order:
        row = rows_by_key.pop((policy, pass_name), None)
        if row is not None:
            rows.append(row)
rows.extend(rows_by_key[key] for key in sorted(rows_by_key))

aggregate_path = base / "aggregate_summary.json"
aggregate_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def pct(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"

def ms(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"

lines = [
    "# ShareGPT 300 HiCache Matrix Summary",
    "",
    "| policy | pass | completed | prefix cache reused | disk prefetch | mean TTFT ms | mean TPOT ms | mean E2E ms | storage keys |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        "| {policy} | {pass_name} | {completed} | {prefix} | {disk} | {ttft} | {tpot} | {e2e} | {keys} |".format(
            policy=row["policy"],
            pass_name=row["pass"],
            completed=row["completed"],
            prefix=pct(row["prefix_cache_reused_ratio"]),
            disk=pct(row["disk_prefetch_ratio"]),
            ttft=ms(row["mean_ttft_ms"]),
            tpot=ms(row["mean_tpot_ms"]),
            e2e=ms(row["mean_e2e_latency_ms"]),
            keys=row["storage_keys"],
        )
    )
matrix_md = base / "aggregate_summary.md"
matrix_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("============ Matrix Aggregate Summary ============")
for row in rows:
    print(
        f"{row['policy']:>4} {row['pass']:<5} "
        f"prefix={pct(row['prefix_cache_reused_ratio']):>8} "
        f"disk={pct(row['disk_prefetch_ratio']):>8} "
        f"ttft={ms(row['mean_ttft_ms']):>8}ms "
        f"tpot={ms(row['mean_tpot_ms']):>8}ms"
    )
print(f"Aggregate JSON: {aggregate_path}")
print(f"Aggregate MD:   {matrix_md}")
print("==================================================")
PY
