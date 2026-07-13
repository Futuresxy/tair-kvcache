# SGLang 0.5.6.post2 多级 KVCache 仿真

## 结论与边界

HiSim 的 HBM/DRAM/SSD 容量、LRU/FIFO、写入/降级/提升、预取成本、缓存读取与重计算决策，
位于框架无关的 `hisim.simulation.tiered_cache`。vLLM 0.23 和 SGLang 0.5.6.post2 使用同一份
策略核心，不各自维护一套算法。

框架适配层仍然必须存在，因为两个框架暴露的控制点不同：vLLM 通过
`KVConnectorBase_V1` 与原生 `SchedulerOutput` 接入；SGLang 通过 `HiRadixCache`、
`HiCacheController` 和 storage backend 接入。适配层只负责把原生 block/page hash、HBM 命中、
异步预取完成量和 write-through 事件翻译成共享核心的输入输出，不能由共享核心直接替代。

| 层 | 负责内容 |
|---|---|
| SGLang 原生控制流 | 请求队列、batch、HBM RadixCache、HiCache 搬运协议、prefill/decode 生命周期 |
| SGLang HiSim 适配器 | 子进程 Hook 安装、page hash 翻译、host staging、部分预取进度、指标落盘 |
| HiSim 共享核心 | DRAM/SSD 容量、LRU/FIFO、连续前缀、提升/降级、持久化、成本决策与统一指标 |
| 性能模型 | 模型真实 KV bytes、DRAM/SSD 有效带宽、固定时延、重计算曲线和 overlap 比例 |

`Instance` 隔离是硬约束：启用多级缓存时必须设置 `instance_id` 或
`HISIM_SGLANG_INSTANCE_ID`；配置与环境不一致会立即报错。namespace 还包含模型、dtype、
page size、TP rank/size 和真实 KV bytes，跨 Instance 或不兼容部署不会命中。

## 请求路径

1. SGLang 原生 `HiRadixCache` 先匹配 HBM。
2. SGLang 的 native host pool 在本模式下仅是搬运暂存区，不作为无限逻辑 DRAM；所有 off-HBM
   页都进入共享策略，避免绕过配置的 DRAM/SSD 容量。
3. 共享核心检查连续 page hash，按 `none`、`always` 或 `cost_aware` 选择读取长度。
4. SGLang `HiCacheController` 按混合 DRAM/SSD 读取成本推进 best-effort 预取，只提交实际完成页；
   未完成页不会被统计为 runtime hit，也不会错误执行 SSD→DRAM 提升。
5. 冷请求结束时，Hook 排空 HBM→host→shared storage 的两段 write-through 队列，使下一轮暖请求
   能看到已写入的 DRAM/SSD 状态。

HBM 容量与逐出继续由 SGLang 的 `max_total_tokens` 和 RadixCache 管理；共享核心管理逻辑 DRAM、
SSD。两层使用相同的 `lru` 或 `fifo` 顺序。这样既复用真实框架调度，又避免把策略散落到两个
框架版本中。

## 策略语义

逐出策略：

- `lru`：命中会刷新访问顺序；
- `fifo`：只按首次写入顺序逐出，命中不改变淘汰顺序。

预取/计算策略：

- `none`：不读取 off-HBM KV，全部重计算；
- `always`：读取全部连续可用前缀，用于最大复用率和 I/O 上界实验；
- `cost_aware`：枚举连续前缀长度，最小化
  `effective_read(prefix) + recompute(remaining)`；收益低于 `min_savings_ms` 时重计算。

重计算模型为：

```text
cost(start, end) = a * (end - start) + b * (end^2 - start^2)
```

其中 `a=recompute_ms_per_token`，`b=recompute_ms_per_token_squared`。二次项用于表达 full
attention 在长上下文中的增长。`prefetch_overlap_fraction` 只影响策略的有效临界路径成本；
SGLang runtime 的部分完成量仍使用原始读取时间和真实可重叠窗口推进。

SGLang 自身的 `hicache_storage_prefetch_policy=best_effort|wait_complete|timeout` 表示“何时结束
异步操作”，不是缓存 admission 策略。它与 HiSim 的 `none|always|cost_aware` 是两个正交维度。

## 配置

示例见 `hisim/experiments/sglang/rtx4090_qwen3_0.6b_tiered_kv_example.json`。可把
`tiered_kv_cache` 放进 `HISIM_CONFIG_PATH`，也可用 `HISIM_SGLANG_TIERED_KV_CONFIG` 指向独立
JSON。示例在 `platform.accelerator` 中内联 RTX 4090 完整规格，因此不依赖修改全局硬件注册表，
干净仓库也可独立复现。主要字段如下：

| 字段 | 含义 |
|---|---|
| `dram_capacity_blocks/bytes/gb` | 逻辑 CPU DRAM 容量 |
| `ssd_capacity_blocks/bytes/gb` | 逻辑 SSD 容量 |
| `eviction_policy` | `lru` 或 `fifo` |
| `prefetch_policy` | `none`、`always` 或 `cost_aware` |
| `write_policy` | `none` 或 `write_through` |
| `*_read/write_bandwidth_gbps` | 端到端有效带宽，不是介质标称带宽 |
| `*_read/write_latency_ms` | 每次触及该层的固定时延 |
| `max_prefetch_blocks` | 单请求最大预取页数，0 表示不额外限制 |
| `ssd_state_path` | SSD 元数据持久化文件，包含并校验 Instance namespace |
| `metrics_path` | 共享核心指标输出 |
| `decision_trace_path` | 逐请求决策 JSONL |

容量换算与传输时间使用模型真实 KV 页大小。Qwen3-0.6B、TP=PP=1、FP16 KV、page size 1 的
当前验证值是 114,688 B/token；Mock tensor 的紧凑物理分配大小不能用于该计算。

## 指标

`metrics.json` 保留统一 TTFT、TPOT、ITL、E2E、吞吐、HBM prefix reuse 和 disk prefetch
指标，并增加 `tiered_kv_cache`：

- `metrics`：策略查询、candidate、计划命中、读写字节、逐出、提升、估算时延；
- `runtime_metrics`：SGLang best-effort 实际完成的 HBM/DRAM/SSD hit tokens、实际命中率和
  `prefetch_completion_rate`；
- `last_decisions`：每请求读缓存/重计算、候选与选择页数、加载/重计算成本和收益；
- `dram/ssd`：容量与当前使用页数。

论文实验报告 runtime hit rate 时应使用 `runtime_metrics.actual_hit_rate`；研究策略本身的选择时
同时报告 `metrics` 和 `last_decisions`。`disk_prefetch_ratio` 是完成预取 token / input token，
与 runtime 指标可交叉校验。

## 复现

固定环境为 conda `hisim`、SGLang 0.5.6.post2。脚本只读取本地模型，不下载权重：

```bash
PYTHONPATH=hisim/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n hisim python \
  hisim/experiments/sglang/validate_tiered_engine.py \
  --model /absolute/path/to/local/Qwen3-0.6B \
  --simulation-config hisim/experiments/sglang/rtx4090_qwen3_0.6b_tiered_kv_example.json \
  --tiered-config hisim/experiments/sglang/tiered_kv_always_validation.json \
  --output /tmp/hisim/sglang-tiered-validation/result.json
```

把 `--tiered-config` 也指向 simulation config 可验证 `cost_aware`。脚本通过真实 SGLang
`Engine`/Scheduler/HiRadixCache 控制流运行冷请求，排空 write-through，清 HBM/native host，再运行
相同暖请求。

当前 RTX 4090 功能验证摘要见
`hisim/experiments/sglang/results/rtx4090_qwen3_0.6b_tiered_validation.json`。示例 predictor 暂用
H100 AIConfigurator 数据库代理，因此它证明控制流、容量、命中与决策闭环，不构成 SGLang 在
RTX 4090 上的最终 TTFT/TPOT 精度校准。论文数据必须再用真实 SGLang 0.5.6.post2 服务测量并拟合
predictor，保留训练/held-out 分离和完整环境元数据。
