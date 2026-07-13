# LLMServingSim ShareGPT 300 HiCache Replay Report

## 1. 实验目标

本实验使用 LLMServingSim 的真实 ShareGPT workload：

```text
/home/songxy/workspace/KVSim-LLM/ref/simulator_sources/LLMServingSim-main/workloads/sharegpt-qwen3-32b-300-sps10.jsonl
```

目标不是 smoke test，而是跑一组完整的 300 请求仿真，并显式开启 SGLang/HiSim 的多级 KVCache 路径：

| 功能 | 本次设置 |
| --- | --- |
| hierarchical KV cache | `--enable-hierarchical-cache` |
| storage backend | `--hicache-storage-backend file` |
| storage prefetch policy | `best_effort` |
| write policy | `write_through` |
| radix eviction policy | `lru` 与 `lfu` 对比 |
| simulated device token pool | `--max-total-tokens 32768` |

`MAX_TOTAL_TOKENS=32768` 用于制造 cache 压力，使 radix cache 淘汰和 HiCache storage prefetch 更容易被触发。当前 workload 最大输入长度为 4570，因此该值不会因为单请求上下文过长导致请求非法。

## 2. 一键复现

从 Hisim 项目根目录运行：

```bash
bash test/scripts/run_llmservingsim_sharegpt_300_hicache_matrix.sh
```

脚本会执行四组实验：

| policy | pass | 含义 |
| --- | --- | --- |
| `lru/cold` | cold | 清空 storage 后运行 LRU |
| `lru/warm` | warm | 复用 `lru/cold` 写入的 storage |
| `lfu/cold` | cold | 清空 storage 后运行 LFU |
| `lfu/warm` | warm | 复用 `lfu/cold` 写入的 storage |

单独跑一组：

```bash
RADIX_EVICTION_POLICY=lru \
HISIM_RESET_HICACHE_STORAGE=1 \
OUT_DIR=test/results/llmservingsim_sharegpt_300_hicache/lru/cold \
bash test/scripts/run_llmservingsim_sharegpt_300_hicache.sh
```

常用覆盖项：

```bash
MAX_TOTAL_TOKENS=65536 \
HICACHE_STORAGE_PREFETCH_POLICY=wait_complete \
HICACHE_STORAGE_BACKEND_EXTRA_CONFIG='{"prefetch_threshold": 512}' \
bash test/scripts/run_llmservingsim_sharegpt_300_hicache_matrix.sh
```

## 3. 输入负载

转换后的 Hisim workload 已放在：

```text
test/assets/llmservingsim/sharegpt-qwen3-32b-300-sps10.hisim.jsonl
```

转换脚本：

```text
test/scripts/llmservingsim_to_hisim.py
```

本次输入统计：

| 指标 | 数值 |
| --- | ---: |
| 请求数 | 300 |
| 输入 tokens 总量 | 258691 |
| 输出 tokens 总量 | 199486 |
| 输入长度平均值 | 862.3 |
| 输入长度 p50 / p90 / max | 729 / 1445 / 4570 |
| 输出长度平均值 | 665.0 |
| 输出长度 p50 / p90 / max | 641 / 785 / 1224 |
| best prior common prefix 平均值 | 70.0 |
| best prior common prefix p50 / p90 / max | 1 / 257 / 1352 |

这份 ShareGPT workload 比 SWE-bench agentic workload 的 prefix 重叠低很多，因此 cold pass 的 cache hit rate 不应期待很高；它更接近自然会话请求分布。

## 4. 模型、硬件和配置

默认仿真配置：

```text
configs/simulation/qwen3_32b_h100_aic.json
```

关键参数：

| 类型 | 当前值 |
| --- | --- |
| model path | `Qwen/Qwen3-32B-FP8` |
| target accelerator | `H100_SXM` |
| predictor | `aiconfigurator / h100_sxm` |
| scheduler dtype | `FP8` |
| KV cache dtype | `FP16` |
| disk read/write bandwidth | `8 GB/s / 8 GB/s` |
| host memory read/write bandwidth | `64 GB/s / 64 GB/s` |
| SGLang runtime device | `cpu` mock |
| SGLang backend version | `0.5.6.post2` |

`device=cpu` 表示本地用 CPU mock server 驱动 SGLang 控制流，不表示目标硬件是 CPU。模型推理耗时由 HiSim 的 AIConfigurator predictor 按 H100_SXM 配置估算。

## 5. 仿真框架

当前路径不是单独重写 serving engine，而是在 SGLang 上安装 HiSim hooks：

1. `hisim.simulation.sglang.launch_server` 安装 hook 后启动 SGLang server。
2. `bench_serving.py` 读取 `hisim-collection` JSONL，把 tokenized prompt 发送给 SGLang。
3. SGLang 保留真实调度、radix prefix cache、HiRadixCache、请求队列等 serving 控制流。
4. HiSim hook 替换模型 forward 和 cache/storage 相关实现，不加载真实权重，而是用 predictor 估算每个 batch 的 forward latency。
5. `StateManager` 推进仿真全局时钟，`sglang_hook.py` 记录每个 request 的 queue、TTFT、TPOT、prefix reuse、storage prefetch。

多级 cache 的层次是：

| 层级 | 说明 |
| --- | --- |
| device KV pool | 由 `max_total_tokens` 限制，本实验为 32768 tokens |
| host HiCache pool | `enable_hierarchical_cache` 后创建，默认按 `hicache_ratio=2.0` 分配 |
| storage backend | 本实验使用 HiSim mock `file` backend，key 文件由 `HISIM_HICACHE_STORAGE_PATH` 指定 |

项目约束要求 KVCache 只在同一个 `instance_id` 内复用。脚本按 policy 隔离 storage 文件：

```text
test/results/llmservingsim_sharegpt_300_hicache/lru/hicache_storage_keys.txt
test/results/llmservingsim_sharegpt_300_hicache/lfu/hicache_storage_keys.txt
```

cold/warm pass 复用同一个 policy 的 storage 文件，用来模拟同一服务实例或同一逻辑 cache storage 的冷启动与热启动对比；LRU 与 LFU 之间不共享 storage。

## 6. 指标计算

核心指标来自 server 侧：

```text
test/results/llmservingsim_sharegpt_300_hicache/<policy>/<pass>/hisim_output/metrics.json
```

`prefix_cache_reused_ratio`：

```text
sum(request.final_reused_tokens) / sum(request.input_length)
```

`final_reused_tokens` 来自 SGLang request 的 `cached_tokens`，反映本次请求进入 prefill 时已经可复用的 prefix KV tokens。

`disk_prefetch_ratio`：

```text
sum(request.prefetch_complete_tokens) / sum(request.input_length)
```

`prefetch_complete_tokens` 是 HiCache storage prefetch 成功加载回 host/cache 路径的 token 数。cold pass storage 为空，因此该值为 0；warm pass 复用 cold 写入的 storage keys，因此会出现非零 disk prefetch。

延迟指标：

| 指标 | 含义 |
| --- | --- |
| TTFT | 从请求进入仿真队列到首 token 完成，包含排队、prefill、首个 decode |
| TPOT | 除首 token 外，每个输出 token 的平均耗时 |
| ITL | 相邻输出 token 间隔 |
| E2E latency | 单请求所有 token latency 之和 |

命令行里部分字段为 `-1`，例如 peak concurrency、vision tokens，是当前 `bench_serving` simulation 模式未实现或不适用的字段，不影响 HiSim server 侧 cache/latency metrics。

`cpu graph: False` 是 CPU mock backend 不启用 CUDA graph 的正常表现，不是建图失败导致的错误。

## 7. 实验结果

汇总文件：

```text
test/results/llmservingsim_sharegpt_300_hicache/aggregate_summary.md
test/results/llmservingsim_sharegpt_300_hicache/aggregate_summary.json
```

本次结果：

| policy | pass | completed | prefix cache reused | disk prefetch | mean TTFT ms | mean TPOT ms | mean E2E ms | storage keys |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | cold | 300 | 4.72% | 0.00% | 64872.36 | 29.24 | 83994.61 | 241024 |
| lru | warm | 300 | 19.74% | 28.93% | 65534.61 | 34.67 | 88155.49 | 243469 |
| lfu | cold | 300 | 4.74% | 0.00% | 64925.02 | 29.25 | 84053.07 | 241023 |
| lfu | warm | 300 | 15.27% | 29.25% | 64176.10 | 34.05 | 86401.07 | 243546 |

所有正式 server log 均检查通过：

| 检查项 | 结果 |
| --- | --- |
| completed requests | 四组均为 300 |
| OOM estimator warnings | 四组均为 0 |
| scheduler exception / traceback / killed | 四组均为 0 |

## 8. 结果分析

cold pass 的 prefix cache reused ratio 只有约 4.7%，原因是 ShareGPT 300 的真实 prefix 重叠本身较低，且本实验把 device token pool 限到 32768 tokens，会引入较强的 radix cache 淘汰压力。

warm pass 的 disk prefetch ratio 达到约 29%，说明 `enable_hierarchical_cache + hicache_storage_backend=file + write_through` 的 storage 写入和下一轮 prefetch 已经生效。warm pass 的 prefix cache reused ratio 也明显高于 cold pass：LRU 从 4.72% 提升到 19.74%，LFU 从 4.74% 提升到 15.27%。

LRU warm 的 prefix cache reuse 高于 LFU warm，但 LFU warm 的 mean TTFT/E2E 略低。原因是整体延迟不仅取决于 prefix hit，还取决于请求到达时间、排队、batch shape、decode 长度、storage prefetch 与计算重叠。不能只用单个 cache ratio 判断策略优劣，需要结合 `request.jsonl` 和 `iteration.jsonl` 看请求级和 iteration 级行为。

warm pass 的 TPOT 高于 cold pass，主要是 storage prefetch 和 host/cache 管理路径加入了额外开销；本实验的 `best_effort` 会尽量在可用推理时间窗口内做 prefetch，而不强制等待所有 storage 数据加载完成。

## 9. 输出文件说明

每组实验目录：

```text
test/results/llmservingsim_sharegpt_300_hicache/<policy>/<pass>/
```

关键文件：

| 文件 | 含义 |
| --- | --- |
| `*.result.jsonl` | `bench_serving` 原始结果，一行 JSONL，包含 benchmark 与 server_info |
| `*.result.pretty.json` | 上述结果的 pretty JSON，方便阅读 |
| `*.summary.json` | 单组实验的精简摘要，脚本生成 |
| `server.log` | SGLang/HiSim server 日志 |
| `hisim_output/metrics.json` | HiSim server 侧聚合指标，pretty JSON |
| `hisim_output/request.jsonl` | request 级统计，含 `final_reused_tokens`、`prefetch_complete_tokens` |
| `hisim_output/iteration.jsonl` | iteration/batch 级统计，含 forward 和 L2 load/backup latency |

## 10. 本次修复

正式 warm pass 首次运行时暴露了一个 HiSim bug：storage prefetch 部分完成时，`calc_prefetch_pages` 返回 float，随后 `operation.completed_tokens` 被 SGLang `hiradix_cache.py` 当作 slice 下标使用，触发：

```text
TypeError: slice indices must be integers or None or have an __index__ method
```

修复点：

| 文件 | 修改 |
| --- | --- |
| `src/hisim/simulation/sglang/sglang_hook.py` | `calc_prefetch_pages` 返回整数，并确保 chunked prefetch 累加后仍是 int |
| `src/hisim/simulation/manager/env.py` | 新增 `HISIM_HICACHE_STORAGE_PATH` 读取 |
| `src/hisim/simulation/sglang/sglang_mock_class.py` | mock storage 不再硬编码 `/tmp/hisim/hicache/storage_keys.txt`，改为脚本指定路径 |

修复后 20 请求 warm 验证和 300 请求完整矩阵均通过。
