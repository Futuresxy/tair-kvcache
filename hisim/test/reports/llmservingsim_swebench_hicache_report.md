# LLMServingSim SWE-bench HiCache Replay Report

## 1. 实验目标

本实验在开启多级 KVCache 卸载的配置下，重新跑 SWE-bench workload：

```text
/home/songxy/workspace/KVSim-LLM/ref/simulator_sources/LLMServingSim-main/workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl
```

该源文件包含 50 个 session、765 个 sub_requests。它不是只有 40 条请求。相比 ShareGPT 300，这个 workload 的上下文重合高很多，更适合验证 prefix cache 与 HiCache。

## 2. 输入数据

已生成两份 Hisim workload：

| 文件 | 请求数 | 用途 |
| --- | ---: | --- |
| `test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-40.hisim.jsonl` | 40 | smoke 对照 |
| `test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-300.hisim.jsonl` | 300 | 更完整的高上下文重合测试 |

300 条转换命令：

```bash
python test/scripts/llmservingsim_to_hisim.py \
  -i /home/songxy/workspace/KVSim-LLM/ref/simulator_sources/LLMServingSim-main/workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl \
  -o test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-300.hisim.jsonl \
  --max-requests 300
```

300 条 workload 的转换统计：

| 指标 | 数值 |
| --- | ---: |
| sessions used | 19 |
| sub_requests | 300 |
| total input tokens | 1848577 |
| total output tokens | 50328 |
| input avg / p50 / p90 / max | 6161.9 / 5864 / 10708 / 21418 |
| output avg / p50 / p90 / max | 167.8 / 112 / 330 / 999 |
| best prior common prefix avg / p50 / p90 / max | 5589.1 / 5318 / 10452 / 18763 |

这解释了为什么 SWE-bench 的 prefix cache hit 可以接近 90%，而 ShareGPT 300 只有十几个到二十个百分点。

## 3. 运行配置

本次使用：

| 参数 | 值 |
| --- | --- |
| model | `Qwen/Qwen3-32B-FP8` |
| sim config | `configs/simulation/qwen3_32b_h100_aic.json` |
| target device spec | `H100_SXM` |
| hierarchical cache | enabled |
| storage backend | `file` |
| write policy | `write_through` |
| prefetch policy | `best_effort` |
| radix eviction policy | `lru` |
| max total tokens | `32768` |

cold/warm 含义：

| pass | 含义 |
| --- | --- |
| cold | 运行前清空 storage，`HISIM_RESET_HICACHE_STORAGE=1` |
| warm | 不清空 storage，复用 cold 写入的 storage keys |

## 4. 复现命令

40 条 cold：

```bash
PORT=30004 NUM_PROMPTS=40 \
DATASET_PATH=test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-40.hisim.jsonl \
OUT_DIR=test/results/llmservingsim_swebench_hicache/40/lru/cold \
RUN_NAME=swe-bench-qwen3-30b-a3b-40-lru-cold \
RADIX_EVICTION_POLICY=lru \
HISIM_HICACHE_STORAGE_PATH=test/results/llmservingsim_swebench_hicache/40/lru/hicache_storage_keys.txt \
HISIM_RESET_HICACHE_STORAGE=1 \
bash test/scripts/run_llmservingsim_sharegpt_300_hicache.sh
```

40 条 warm：

```bash
PORT=30005 NUM_PROMPTS=40 \
DATASET_PATH=test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-40.hisim.jsonl \
OUT_DIR=test/results/llmservingsim_swebench_hicache/40/lru/warm \
RUN_NAME=swe-bench-qwen3-30b-a3b-40-lru-warm \
RADIX_EVICTION_POLICY=lru \
HISIM_HICACHE_STORAGE_PATH=test/results/llmservingsim_swebench_hicache/40/lru/hicache_storage_keys.txt \
HISIM_RESET_HICACHE_STORAGE=0 \
bash test/scripts/run_llmservingsim_sharegpt_300_hicache.sh
```

300 条 cold/warm 只需把 `NUM_PROMPTS` 和 `DATASET_PATH/OUT_DIR/RUN_NAME/HISIM_HICACHE_STORAGE_PATH` 改为 `300` 对应路径。

## 5. 实验结果

汇总文件：

```text
test/results/llmservingsim_swebench_hicache/aggregate_summary.md
test/results/llmservingsim_swebench_hicache/aggregate_summary.json
```

本次结果：

| scale | pass | completed | prefix reuse | disk prefetch | mean TTFT ms | mean TPOT ms | mean E2E ms | L2 load sum s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | cold | 40 | 91.37% | 0.00% | 68.78 | 22.73 | 5330.51 | 0.00 |
| 40 | warm | 40 | 94.89% | 7.03% | 50.15 | 22.16 | 5187.42 | 0.03 |
| 300 | cold | 300 | 89.89% | 0.00% | 126007.19 | 56.15 | 133720.05 | 143.84 |
| 300 | warm | 300 | 62.49% | 3.52% | 92448.52 | 34.41 | 97537.25 | 21.16 |

所有 server log 均检查通过：

| 检查项 | 结果 |
| --- | --- |
| completed | 40/40 与 300/300 |
| OOM estimator warnings | 0 |
| scheduler exception / traceback / killed | 0 |

## 6. 结果解释

40 条 smoke 的 cold pass 已经达到 91.37%，说明之前看到的 91% 没有消失。该高命中率来自 SWE-bench agentic session 的上下文复用，而不是偶然结果。

40 条 warm pass 进一步到 94.89%，并出现 7.03% disk prefetch，TTFT 从 68.78 ms 降到 50.15 ms。

300 条 cold pass 也保持 89.89% prefix reuse，说明更完整的 SWE-bench 采样同样具备强 prefix 重合。

300 条 warm pass 的 `prefix_cache_reused_ratio` 降到 62.49%，但 TTFT/TPOT/E2E 明显改善。原因是该指标只统计请求进入 prefill 时最终复用的 prefix tokens，不等价于总性能收益。warm pass 中 host/storage 迁移行为改变了 radix cache 保留结构，导致最终 prefix reuse ratio 下降；同时 L2 load 总耗时从 143.84s 降到 21.16s，减少了大量 host load-back 等待，因此整体延迟更好。

因此这组结果应同时看：

| 指标 | 解释 |
| --- | --- |
| `prefix_cache_reused_ratio` | 进入 prefill 时的有效 prefix KV 复用比例 |
| `disk_prefetch_ratio` | storage 成功预取比例 |
| `L2 load sum` | host/device 层 KV 迁移总成本 |
| TTFT/TPOT/E2E | 端到端性能结果 |

不能只用单个 prefix hit rate 判断 HiCache 是否有效。
