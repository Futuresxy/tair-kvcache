# LLMServingSim SWE-bench Workload Replay in Hisim

> 说明：本文记录 40 条 SWE-bench smoke replay。300 条 ShareGPT + HiCache cold/warm + LRU/LFU 的正式实验见 `test/reports/llmservingsim_sharegpt_300_hicache_report.md`。

## 1. 实验目标

本实验验证一份真实长上下文 workload 在 Hisim + SGLang mock simulation 路径下的 KVCache 复用效果和延迟指标。

本次 smoke workload 规模较小，目标是确认链路正确、指标可复现，而不是做完整吞吐压力测试。

## 2. 一键复现

从 Hisim 项目根目录运行：

```bash
bash test/scripts/run_llmservingsim_hisim_smoke.sh
```

脚本会自动完成：

1. 使用已转换好的 40 条 SWE-bench Hisim workload。
2. 使用默认仿真配置 `configs/simulation/qwen3_32b_h100_aic.json`。
3. 启动 CPU SGLang mock server。
4. 用 `bench_serving` 发送 `hisim-collection` 请求。
5. 在 server 侧触发 Hisim offline simulation。
6. 输出 benchmark 结果和 Hisim server 侧 metrics。

可覆盖配置：

```bash
SIM_CONFIG=/path/to/config.json bash test/scripts/run_llmservingsim_hisim_smoke.sh
```

可覆盖输入 workload：

```bash
PRECONVERTED_DATASET=/path/to/workload.hisim.jsonl \
  NUM_PROMPTS=40 \
  bash test/scripts/run_llmservingsim_hisim_smoke.sh
```

## 3. 输入负载

当前输入文件：

```text
test/assets/llmservingsim/swe-bench-qwen3-30b-a3b-40.hisim.jsonl
```

这份 workload 来自 LLMServingSim 中 SWE-bench agentic trace 的 40 条采样。它保留了真实 token ids 和多轮 agent 上下文累积，因此天然包含大量历史前缀重叠，适合验证 prefix KVCache reuse。

每行是一个 Hisim `hisim-collection` 请求，关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `rid` | 请求 id |
| `timestamp` | 请求到达时间，单位秒 |
| `input_ids` | 已 tokenized 的输入 token ids |
| `input_length` | 输入 token 数 |
| `output_length` | 需要生成的 token 数 |
| `output_ids` | 原始 trace 中的输出 token ids，仅作为 workload 记录 |

转换脚本：

```text
test/scripts/llmservingsim_to_hisim.py
```

LLMServingSim 到 Hisim 的核心映射：

| LLMServingSim | Hisim |
| --- | --- |
| `arrival_time_ns / 1e9` | `timestamp` |
| `input_tok_ids` | `input_ids` |
| `len(input_tok_ids)` | `input_length` |
| `output_toks` | `output_length` |
| `output_tok_ids` | `output_ids` |

Agentic workload 的 `sub_requests` 会被 flatten 成多条 Hisim 请求。当前 replay 没有显式 `instance_id` 字段，因此一次脚本运行对应单个 SGLang/Hisim cache instance。按项目约束，KVCache 只应在同一个 instance 内复用，不能跨 instance 匹配。

## 4. 模型、硬件和仿真配置

默认配置文件：

```text
configs/simulation/qwen3_32b_h100_aic.json
```

关键配置：

```json
{
  "platform": {
    "accelerator": {
      "name": "H100_SXM"
    },
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 8
  },
  "predictor": {
    "name": "aiconfigurator",
    "device_name": "h100_sxm"
  },
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "data_type": "FP8",
    "kv_cache_data_type": "FP16",
    "backend_version": "0.5.6.post2"
  }
}
```

Hisim 中有两类配置，不要混淆：

| 类型 | 位置 | 作用 |
| --- | --- | --- |
| 硬件 spec | `src/hisim/spec/accelerator/info.py` | 注册 H100、H20、RTX4090、Ascend910B 等硬件事实，用于 KVCache 容量和带宽估算 |
| 模型 spec | `src/hisim/spec/model/info.py` | 注册 Qwen、DeepSeek 等常见模型作为离线 fallback |
| 仿真场景 config | `configs/simulation/*.json` | 选择本次实验使用哪个硬件、predictor database、TP/EP/DP、dtype |

运行 SGLang server 时，如果 SGLang 能拿到 HuggingFace config，Hisim 会优先解析运行时模型 config；内置模型 spec 主要用于离线 fallback 和测试。

本次 server 运行结果中关键 runtime 参数：

| 参数 | 当前值 |
| --- | --- |
| Model | `Qwen/Qwen3-32B-FP8` |
| Platform spec | `H100_SXM` |
| AIC predictor system | `h100_sxm` |
| SGLang device | `cpu` |
| Attention backend | `torch_native` |
| TP / EP / DP | `1 / 1 / 1` |
| Simulated KV token capacity | `160561` |

`device=cpu` 是 mock server 的运行方式，不代表仿真目标是 CPU。目标硬件由 Hisim config 和 AIConfigurator predictor 决定。

## 5. Hisim 与 SGLang 的关系

当前仿真器不是单独重写一个 serving engine，而是在 SGLang 上安装 hook：

1. `hisim.simulation.sglang.launch_server` 启动前安装 SGLang hooks。
2. SGLang 负责请求协议、tokenized request、radix prefix cache、调度队列等 serving 结构。
3. Hisim 替换或拦截部分 SGLang 组件，使其不跑真实模型权重，而是用性能模型估算每个 batch 的耗时。
4. Hisim 用 `StateManager` 维护一个仿真全局时钟，并在每次 SGLang batch 完成后推进时间。
5. Hisim 从 SGLang request/batch 对象中读取 `cached_tokens`、prefix length、decode/prefill batch shape，统计 cache hit 和延迟。

核心模块：

| 模块 | 作用 |
| --- | --- |
| `bench_serving.py` | benchmark client，读取 workload 并发送请求 |
| `launch_server.py` | 安装 Hisim hooks 后启动 SGLang server |
| `sglang_hook.py` | scheduler/model/cache hooks，统计请求、调用 predictor、推进仿真时钟 |
| `aiconfigurator.py` | AIConfigurator latency predictor 封装 |
| `simulation/utils.py` | 聚合 TTFT/TPOT/cache ratio 等 metrics |

当前 smoke 默认使用 offline simulation。请求会先全部进入 future queue，Hisim 按 trace timestamp 推进全局时钟并释放到 SGLang scheduler。

## 6. Cache Hit Rate 如何得到

本实验报告的 cache hit 指标是：

```text
prefix_cache_reused_ratio = sum(final_reused_tokens) / sum(input_length)
```

数据来源：

1. SGLang radix prefix cache 在 prefill 前会计算本次请求能复用多少历史 prefix token。
2. Hisim 在 `get_new_batch_prefill` hook 中读取每个请求的 `req.cached_tokens`。
3. 该值写入 `request.jsonl` 的 `final_reused_tokens`。
4. `calc_metrics()` 聚合所有请求，得到 `prefix_cache_reused_ratio`。

当前结果：

```text
sum(final_reused_tokens) = 248794
sum(input_length)        = 272298
prefix_cache_reused_ratio = 248794 / 272298 = 0.9136828034
```

这表示输入 token 中约 91.37% 命中了 prefix cache，可以跳过对应的 prefill KV 计算。

注意：这个指标是 prefix cache reuse，不是二级存储 prefetch 命中率。二级存储相关指标是 `disk_prefetch_ratio`。

## 7. 延迟指标如何得到

Hisim 在每个 SGLang batch 运行后构造 `FakeRequest`，把 batch shape 交给 AIConfigurator predictor：

- Prefill/extend batch：使用 `input_length` 和 `past_kv_length`。
- Decode batch：每个请求的 `input_length=1`，使用当前 `past_kv_length`。

AIConfigurator 返回当前 batch 的预测耗时，Hisim 将其记录为 `current_inference_dur`，并推进 `StateManager` 的全局仿真时钟。

每次 batch 处理结果时，Hisim 计算：

```text
one_token_latency = request_response_time - request.last_event_time
```

并追加到该请求的 `gen_token_latencies`。因此：

| 指标 | 计算方式 |
| --- | --- |
| TTFT | `gen_token_latencies[0]` |
| TPOT | `mean(gen_token_latencies[1:])`，按请求先算平均，再对请求聚合 |
| ITL | 所有请求的 `gen_token_latencies[1:]` 展平后的分布 |
| E2E latency | `sum(gen_token_latencies)` |
| throughput | token/request 总量除以仿真 duration |

如果开启 HiCache L2/load-back，batch latency 还会加上：

```text
hicache_l2_load_dur + current_inference_dur + hicache_l2_backup_dur
```

当前 smoke 未启用多级 KVCache offload，因此 L2 load/backup 都是 0。

## 8. 多级 KVCache、卸载和替换策略

当前 smoke 的 server args 中：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `enable_hierarchical_cache` | `false` | 未开启 SGLang HiCache 多级缓存 |
| `hicache_storage_backend` | `null` | 未开启远端/磁盘 storage backend |
| `hicache_write_policy` | `write_through` | 仅为默认参数；本次未启用 storage backend |
| `disk_prefetch_ratio` | `0.0` | 没有二级存储 prefetch 命中 |
| `l2_load_latency` | 全部为 `0` | 没有 host/storage load |
| `l2_backup_latency` | 全部为 `0` | 没有 backup/offload |
| `disable_radix_cache` | `false` | SGLang radix prefix cache 开启 |
| `radix_eviction_policy` | `lru` | 设备侧 radix cache 淘汰策略为 LRU |

因此，本次 91.37% 的 cache hit 来自 SGLang radix prefix cache 的设备侧前缀复用，不来自多级 KVCache 卸载或磁盘预取。

Hisim 已经 hook 了 HiCache 相关路径。如果后续通过 SGLang 参数开启 `--enable-hierarchical-cache` 并设置 storage backend，Hisim 会：

1. 用 `platform.memory_*_bandwidth_gb` 估算 host/device KV 传输。
2. 用 `platform.disk_*_bandwidth_gb` 估算 storage prefetch/backup。
3. 在 `iteration.jsonl` 中记录 `l2_load_latency` 和 `l2_backup_latency`。
4. 在 `request.jsonl` 中记录 `prefetch_complete_tokens`。
5. 在 `metrics.json` 中得到非零 `disk_prefetch_ratio`。

## 9. 输出文件说明

脚本默认输出目录：

```text
test/results/llmservingsim_swebench/
```

重要文件：

| 文件 | 作用 |
| --- | --- |
| `server.log` | SGLang/Hisim server 日志，包含配置、warning、仿真完成状态 |
| `swe-bench-qwen3-30b-a3b-40.result.jsonl` | benchmark client 输出，一行 JSON |
| `swe-bench-qwen3-30b-a3b-40.result.pretty.json` | 上述 JSON 的 pretty 版本 |
| `hisim_output/metrics.json` | Hisim server 侧聚合指标，已格式化为多行 JSON |
| `hisim_output/request.jsonl` | 每个请求一行，包含每条请求的 cache reuse 和 token latency |
| `hisim_output/iteration.jsonl` | 每个调度 iteration 一行，包含 batch shape、forward latency、L2 load/backup latency |

推荐查看顺序：

1. `hisim_output/metrics.json`：看总览。
2. `hisim_output/request.jsonl`：分析哪条请求贡献了 cache hit。
3. `hisim_output/iteration.jsonl`：分析 batch 级 latency 和调度行为。
4. `server.log`：排查异常、OOM estimator warning、CPU graph 状态等。

## 10. 当前实验结果

当前 smoke 结果：

| Metric | Value |
| --- | ---: |
| Completed requests | 40 / 40 |
| Total input tokens | 272,298 |
| Total output tokens | 9,285 |
| Prefix cache reused tokens | 248,794 |
| Prefix cache reused ratio | 0.9136828034 |
| Disk prefetch ratio | 0.0 |
| Mean TTFT | 67.14 ms |
| Median TTFT | 49.82 ms |
| P99 TTFT | 263.30 ms |
| Mean TPOT | 22.74 ms |
| Median TPOT | 22.30 ms |
| P99 TPOT | 27.83 ms |
| Mean ITL | 22.77 ms |
| P99 ITL | 66.36 ms |
| Mean E2E latency | 5,330.51 ms |
| Simulated duration | 34.83 s |
| Request throughput | 1.15 req/s |

正确性检查：

| Check | Result |
| --- | --- |
| `Traceback` / `ERROR` in server log | 0 |
| AIConfigurator OOM warnings | 0 |
| Negative token latencies | 0 |
| Negative forward latencies | 0 |
| L2 load latency sum | 0 |
| L2 backup latency sum | 0 |

结果解读：

- Prefix cache reuse 很高，原因是 SWE-bench agentic trace 中后续请求携带了大量历史上下文，前缀重复度高。
- `disk_prefetch_ratio=0.0` 是预期行为，因为当前 smoke 未启用 HiCache storage/offload。
- `cpu graph: False` 是预期行为，因为 smoke 脚本用 `--device cpu` 启动 mock server，不使用 CUDA graph。
- benchmark 表格中的部分 `-1` 是通用客户端字段缺省值，不影响 Hisim server 侧 metrics。simulation 模式应以 `hisim_output/metrics.json` 为准。
