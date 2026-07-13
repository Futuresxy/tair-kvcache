# HiSim 架构学习、实验报告优化与 vLLM 接入评估

> 说明：本文基于当前工作区源码与 `hisim/test/reports` 既有实验报告整理。当前工作区不是干净上游状态，本文描述的是本地当前实现与已有测试结果。路径均为仓库相对路径。

## 1. 核心结论

| 主题 | 结论 |
| --- | --- |
| HiSim 定位 | CPU 侧 LLM serving 仿真器，通过 hook 推理框架控制流，避免真实 GPU 模型执行，并用性能模型估算 TTFT、TPOT、吞吐和 KVCache/HiCache 指标。 |
| 当前稳定后端 | 当前 HiSim 的已实现后端是 SGLang。入口、调度、模型执行、HiCache storage 都是 SGLang 内部类 hook。 |
| 当前稳定场景 | README 与配置覆盖 Qwen3 8B/32B、H20/H100、AIConfigurator predictor；实验报告覆盖 LLMServingSim 的 ShareGPT 与 SWE-bench replay。 |
| 架构核心 | `bench_serving` 发送 trace 请求，SGLang 负责 serving 控制流，HiSim hook 替换模型/KV 组件，`StateManager` 推进仿真时钟，`AIConfiguratorTimePredictor` 预测 batch 耗时。 |
| vLLM 接入判断 | Manager/Connector 层已有 vLLM connector；HiSim 仿真层尚未接入 vLLM。可接入，但不是改一个 backend flag，而是需要新增 vLLM backend adapter/hook。 |
| Instance 隔离 | 项目约束要求 KVCache 只在同一 `instance_id` 内复用。当前 HiSim mock storage 对多 instance 支持不足，接 vLLM 前必须补齐 instance namespace。 |

## 2. 学习路线

建议按下面顺序学习，而不是直接从 990 行的 `sglang_hook.py` 开始：

1. 先读项目定位：`README_zh.md`、`hisim/README_zh.md`、`hisim/configs/README.md`。
2. 跑通 smoke：`hisim/test/scripts/run_llmservingsim_hisim_smoke.sh`。
3. 看输入数据：`hisim/test/assets/llmservingsim/*.hisim.jsonl` 和 `hisim/test/scripts/llmservingsim_to_hisim.py`。
4. 看 client 侧：`hisim/src/hisim/simulation/bench_serving.py`，重点是 `hisim-collection` 数据集和 `custom_params.simulation`。
5. 看 server 入口：`hisim/src/hisim/simulation/sglang/launch_server.py`，理解 hook 安装顺序。
6. 看核心 hook：`hisim/src/hisim/simulation/sglang/sglang_hook.py`，先看 `C_ModelRunnerHook` 和 `C_SchedulerHook`，再看 HiCache hook。
7. 看性能模型：`hisim/src/hisim/time_predictor/base.py` 和 `hisim/src/hisim/time_predictor/aiconfigurator.py`。
8. 看结果指标：`hisim/src/hisim/simulation/utils.py` 和 `hisim_output/{metrics.json,request.jsonl,iteration.jsonl}`。
9. 最后读现有实验报告：`hisim/test/reports/llmservingsim_*.md`。

## 3. 仓库与模块地图

| 路径 | 作用 | 关键点 |
| --- | --- | --- |
| `hisim/README.md`, `hisim/README_zh.md` | 用户入口文档 | 安装、Quick Start、支持矩阵、配置格式、准确率示例。 |
| `hisim/configs/simulation/*.json` | 仿真场景配置 | 选择目标硬件、AIC database、TP/EP/DP、dtype、backend version。 |
| `hisim/src/hisim/hook/` | 通用 hook 机制 | 通过 `sys.meta_path` 和 `builtins.__build_class__` 改写模块/类加载。 |
| `hisim/src/hisim/simulation/sglang/launch_server.py` | HTTP server 仿真入口 | 安装 SGLang hooks 后调用 SGLang `launch_server`。 |
| `hisim/src/hisim/simulation/sglang/sglang_hook.py` | SGLang 核心适配层 | 替换 model runner、scheduler、HiCache controller、storage backend。 |
| `hisim/src/hisim/simulation/sglang/sglang_mock_class.py` | mock memory/cache/storage | CPU mock pool、host/device 传输耗时估算、mock HiCache storage。 |
| `hisim/src/hisim/time_predictor/` | batch latency predictor | 将 prefill/decode batch 转成 AIConfigurator runtime config。 |
| `hisim/src/hisim/spec/` | 模型/硬件事实注册 | 模型结构、硬件容量/带宽、dtype 信息。 |
| `hisim/src/hisim/dataset/` | Hisim 内部数据集抽象 | `GenericRequest`、随机数据、Hisim collection 数据。 |
| `hisim/src/hisim/simulation/bench_serving.py` | benchmark client | 从 trace/synthetic dataset 发送请求，simulation 模式读取 server 侧 metrics。 |
| `hisim/test/scripts/` | 实验脚本 | LLMServingSim 转换、smoke、ShareGPT/HiCache 矩阵。 |
| `hisim/test/reports/` | 实验报告 | 已有 replay/HiCache 结果，本报告补总体架构与评估。 |

## 4. 架构总览

```mermaid
flowchart TD
  A[LLMServingSim / random / hisim-collection workload] --> B[bench_serving.py]
  B -->|SGLang generate API + custom_params.simulation| C[SGLang server]
  C --> D[HiSim class/module hooks]
  D --> E[Mock ModelRunner / Mock KV Pools]
  D --> F[SGLang Scheduler + Radix/HiRadix control flow]
  F --> G[ScheduleBatch(input_length, past_kv_length)]
  G --> H[AIConfiguratorTimePredictor]
  H --> I[StateManager global clock]
  D --> J[Mock HiCache storage + bandwidth model]
  I --> K[hisim_output metrics/request/iteration]
```

HiSim 当前没有重写 serving engine。它保留 SGLang 的请求队列、调度、radix prefix cache、HiRadixCache/HiCache 控制流，再用 hook 替换模型执行和部分 cache/storage 操作。这样可以用较低成本观察接近真实框架行为的 queue、batch shape、prefix reuse 和多级 KVCache 迁移。

## 5. 核心运行链路

### 5.1 Server 启动

`hisim/src/hisim/simulation/sglang/launch_server.py` 在导入 SGLang server 前安装 hook：

- CPU 环境下安装 `M_SGLangKernelLoadUtilHook`，见 `launch_server.py:12-15`。
- 安装 `C_SchedulerHook`、`C_ModelRunnerHook`、`C_TokenizerManagerHook`、`C_StorageBackendFactory`、`C_HiCacheController`、`C_HiRadixCacheHook`，见 `launch_server.py:16-25`。
- 解析 SGLang 参数和 HiSim `SimulationArgs`，然后写入或读取 `HISIM_CONFIG_PATH`，见 `launch_server.py:45-68`。
- 最后调用 SGLang 的 `launch_server(server_args)`，见 `launch_server.py:70-73`。

### 5.2 Hook 机制

`hisim/src/hisim/hook/module_hook_entry.py` 用 `sys.meta_path` 插入自定义 finder/loader，在目标模块加载时执行原模块代码后调用 hook，见 `module_hook_entry.py:16-20`、`module_hook_entry.py:28-50`、`module_hook_entry.py:60-62`。

`hisim/src/hisim/hook/class_hook_entry.py` 改写 `builtins.__build_class__`，在目标类创建时替换或修改类对象，见 `class_hook_entry.py:18-48`、`class_hook_entry.py:51-53`。这使 HiSim 能在不 fork SGLang 源码的情况下拦截内部类。

### 5.3 请求进入仿真

client 侧 `bench_serving.py` 在 SGLang generate payload 中加入：

```json
"custom_params": {"simulation": request_func_input.simulation}
```

对应代码在 `bench_serving.py:552-567`。离线 replay 时，`get_request()` 会把 trace timestamp 写入 `simulation.created_time` 和 `simulation.total_request`，见 `bench_serving.py:1763-1807`。`hisim-collection` 数据集会按输入 timestamp 排序和重放，见 `bench_serving.py:2184-2187`。

server 侧 `C_TokenizerManagerHook` 会把 blocking 模式的 server 创建时间补回 simulation 参数，见 `sglang_hook.py:59-79`。

### 5.4 离线调度与全局时钟

`C_SchedulerHook` 维护 `FUTURE_QUEUE`、`REQUEST_STATS`、`ITERATION_STATS` 和 `StateManager`：

- OFFLINE 模式先收齐所有请求，再按 trace 时间释放给 SGLang scheduler，见 `sglang_hook.py:640-724`。
- 请求进入 scheduler 时记录 `created_time`、`queue_start`、`input_length`、`output_length`，见 `sglang_hook.py:725-759`。
- prefill batch 形成后读取 SGLang request 的 `cached_tokens`，写入 `final_reused_tokens`，见 `sglang_hook.py:761-793`。
- batch 执行时将 SGLang batch 转成 HiSim `FakeRequest(input_length, past_kv_length)`，见 `sglang_hook.py:795-850`。
- batch 完成后推进全局时钟并追加 token latency，见 `sglang_hook.py:854-908`。
- profile 触发时写出 `metrics.json`、`iteration.jsonl`、`request.jsonl`，见 `sglang_hook.py:910-982`。

`StateManager` 是简单的全局状态机，记录 iteration、global clock、当前/上次 inference 时长、L2 load/backup 时长，见 `state.py:1-69`。

### 5.5 模型执行替换

`C_ModelRunnerHook` 替换 SGLang `ModelRunner.initialize`：

- 不加载真实模型，创建 `MockModel`，见 `sglang_hook.py:90-100`。
- 从 HF runtime config 和 HiSim config 解析 `ModelInfo`、`AcceleratorInfo`、`SchedulerConfig`，见 `sglang_hook.py:102-110`。
- 估算或读取 `max_total_num_tokens`，并创建 mock req/token KV pool，见 `sglang_hook.py:112-190`。
- 替换 `forward`，返回空 logits 结构；替换 `sample`，固定生成 token id，见 `sglang_hook.py:212-258`。

这一步是 HiSim 能在 CPU 上跑 SGLang 控制流的关键。

### 5.6 性能预测

HiSim 的 predictor 抽象很小：

- `FakeRequest` 只包含 `input_length` 和 `past_kv_length`，见 `time_predictor/base.py:12-16`。
- `ScheduleBatch` 判断 prefill/decode，并计算 batch size、context tokens 和 request info，见 `time_predictor/base.py:18-72`。
- `InferTimePredictor.predict_infer_time()` 返回秒级 latency，见 `time_predictor/base.py:74-90`。

`AIConfiguratorTimePredictor` 做两件事：

1. 把 HiSim dtype、模型结构、TP/EP/DP 转成 AIConfigurator 模型和 runtime config，见 `aiconfigurator.py:267-359`。
2. 对 decode 使用平均 past KV length，对 prefill 使用平均 `past_kv + input` 和 `prefix`，调用 AIConfigurator static mode 得到 latency，再乘以 scale factor，见 `aiconfigurator.py:447-518`。

因此，HiSim 的精度主要依赖三个输入：模型结构、目标硬件/数据库、batch shape 抽象是否贴近真实框架。

## 6. KVCache 与 HiCache 仿真

### 6.1 Prefix cache 指标

`prefix_cache_reused_ratio` 的定义在 `calc_metrics()` 中：

```text
sum(final_reused_tokens) / sum(input_length)
```

源码见 `hisim/src/hisim/simulation/utils.py:53-96`。`final_reused_tokens` 来自 SGLang request 的 `cached_tokens`，记录位置见 `sglang_hook.py:761-768`。

注意：这个指标表示进入 prefill 时已有 prefix KV 可复用比例，不等于 storage prefetch ratio，也不能单独代表端到端性能。

### 6.2 HiCache storage 和 L2 迁移

`C_StorageBackendFactory` 将 SGLang storage backend 替换成 `MockHiCacheStorage`，见 `sglang_hook.py:555-565`。

`C_HiCacheController` 接管 backup/prefetch 线程，让操作在调度路径中被显式处理：

- prefetch 耗时按 `required_pages * page_size_byte / disk_read_bandwidth` 估算，见 `sglang_hook.py:269-279`。
- best-effort prefetch 会在当前 inference 时长窗口内完成一部分或全部 storage load，见 `sglang_hook.py:310-415`。
- HiRadixCache 初始化时替换 host KV pool、创建 cache controller、保留 write/load/prefetch 状态，见 `sglang_hook.py:448-552`。

host/device KV 迁移耗时由 `MockTokenToKVPoolHost` 估算：

- H2D load 根据 `memory_read_bandwidth_gb` 估算并累计到 `hicache_l2_load_dur`，见 `sglang_mock_class.py:890-936`。
- D2H backup 根据 `memory_write_bandwidth_gb` 估算并累计到 `hicache_l2_backup_dur`，见 `sglang_mock_class.py:938-963`。

`MockHiCacheStorage` 有两个后端：

- 如果 `kvcm_py_optimizer` 可用，则使用 KVCM optimizer manager 做 storage 命中/写入，见 `sglang_mock_class.py:1024-1029`、`sglang_mock_class.py:1048-1072`。
- 否则使用本地 set + `HISIM_HICACHE_STORAGE_PATH` 文件记录 keys，见 `sglang_mock_class.py:1030-1046`、`sglang_mock_class.py:1084-1164`。

当前一个重要限制是：KVCM optimizer 路径里写死了单个 `instance_id`，并注释说明 multi-instance 尚未支持，见 `sglang_mock_class.py:1071-1072`。这与项目级约束相冲突，后续必须改成从配置或请求上下文传入。

## 7. 配置与可复现路径

### 7.1 配置结构

HiSim 配置分三层：

| 配置 | 来源 | 说明 |
| --- | --- | --- |
| 模型结构 | SGLang runtime HF config 优先，`hisim/src/hisim/spec/model` fallback | `ConfigManager.get_model_info()` 见 `config.py:31-43`。 |
| 硬件事实 | `hisim/src/hisim/spec/accelerator` | `ConfigManager.get_accelerator_info()` 见 `config.py:45-59`。 |
| 仿真场景 | `hisim/configs/simulation/*.json` | `platform`、`predictor`、`scheduler`。 |

`ConfigManager` 当前只支持解析 SGLang server args；其他 backend 会直接报错，见 `config.py:163-178`。这是 vLLM 仿真尚未接入的直接证据。

### 7.2 pytest 与 conda 环境

当前推荐使用 conda `hisim` 环境运行测试和脚本：

```bash
conda run -n hisim python -m pip install pytest
conda run -n hisim python -m pytest hisim/test/test_llmservingsim_conversion.py
```

本地已验证 `pytest 9.1.1` 可用，`hisim/test/test_llmservingsim_conversion.py` 通过。

### 7.3 H20_AIC.zip 应解压到哪里

`download_path` 是 README 和配置文件里的占位目录，不是固定系统路径。当前 H20 配置文件写的是：

```json
"database_path": "download_path/Hisim/Data/aic"
```

Qwen3-8B H20 配置还会读取：

```json
"xgb_model_path": "download_path/Hisim/Data/aic/xgb_models/qwen3_8B"
```

这份 `H20_AIC.zip` 的压缩包顶层目录是 `aic/`。如果不修改配置文件，需要把它解压到 `hisim/download_path/Hisim/Data`，让最终路径正好变成 `hisim/download_path/Hisim/Data/aic`：

```bash
cd /home/songxy/workspace/KVSim-LLM/ref/simulator_sources/tair-kvcache/hisim
mkdir -p download_path/Hisim/Data
unzip test/assets/H20_AIC.zip -d download_path/Hisim/Data
```

解压后应能看到类似路径：

```text
download_path/Hisim/Data/aic
download_path/Hisim/Data/aic/xgb_models/qwen3_8B
```

如果你习惯从仓库根目录或任意目录启动脚本，建议把 `hisim/configs/simulation/qwen3_32b_h20_aic.json` 和 `hisim/configs/simulation/qwen3_8b_h20_aic.json` 中的 `database_path`、`xgb_model_path` 改成绝对路径，避免相对路径随当前工作目录变化。

### 7.4 H20_AIC.zip 是什么

`AIConfiguratorTimePredictor` 会调用 AIConfigurator 的 performance database 来估算 prefill/decode batch latency，见 `aiconfigurator.py:382-414` 和 `aiconfigurator.py:447-518`。`H20_AIC.zip` 就是让 AIConfigurator 认识 H20 这类目标硬件并能查到算子 latency 的本地数据库包。

这里的两个说法可以这样理解：

| 名称 | 含义 | 在 HiSim 中的作用 |
| --- | --- | --- |
| 硬件配置信息 | AIConfigurator 对目标系统的描述，例如 `h20_sxm` 这个 system name、硬件能力、后端版本相关 metadata。 | 让 `predictor.device_name = "h20_sxm"` 能被 `get_database(system=..., backend=..., version=...)` 找到。 |
| 算子插值数据包 | 真实或离线采集的算子 profiling 数据点，以及用于在不同 batch size、sequence length、prefix length 间插值/近似的数据库。 | 当 HiSim 传入 `batch_size`、`isl`、`prefix`、`osl` 时，AIConfigurator 用这些数据估算当前 prefill/decode 耗时。 |

因此，`H20_AIC.zip` 不是模型权重，也不是请求 trace。它是性能预测数据库：没有它，H20 配置下的 `aiconfigurator` predictor 可能找不到 `h20_sxm` 系统或缺少算子延迟数据，只能改用别的 proxy device，预测精度会下降。

### 7.5 最小搭建命令

从 `hisim/` 目录安装：

```bash
cd hisim
pip install .
```

CPU mock server 常用环境变量：

```bash
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

启动 server：

```bash
python3 -m hisim.simulation.sglang.launch_server \
  --model-path Qwen/Qwen3-32B-FP8 \
  --sim-config-path configs/simulation/qwen3_32b_h100_aic.json \
  --skip-server-warmup \
  --device cpu
```

发送 replay：

```bash
python3 -m hisim.simulation.bench_serving \
  --warmup-requests 0 \
  --bench-mode simulation \
  --backend sglang \
  --model Qwen/Qwen3-32B-FP8 \
  --dataset-name hisim-collection \
  --dataset-path test/assets/llmservingsim/sharegpt-qwen3-32b-300-sps10.hisim.jsonl \
  --num-prompts 300 \
  --tokenize-prompt
```

更推荐先跑脚本：

```bash
bash hisim/test/scripts/run_llmservingsim_hisim_smoke.sh
bash hisim/test/scripts/run_llmservingsim_sharegpt_300_hicache_matrix.sh
```

## 8. 已有报告的稳定结论与优化解读

现有三份报告分别解决不同问题：

| 报告 | 价值 | 应如何阅读 |
| --- | --- | --- |
| `llmservingsim_hisim_replay_report.md` | 40 条 SWE-bench smoke，验证链路正确。 | 先看这个，理解 request/iteration/metrics 三类输出。 |
| `llmservingsim_sharegpt_300_hicache_report.md` | 300 条 ShareGPT + LRU/LFU + cold/warm，对比自然会话 workload。 | 用它理解 warm storage、disk prefetch、LRU/LFU 差异。 |
| `llmservingsim_swebench_hicache_report.md` | SWE-bench 长上下文 workload，高 prefix overlap。 | 用它理解 agentic trace 下 prefix reuse 与 L2 load 的关系。 |

### 8.1 ShareGPT 300 结论

汇总见 `hisim/test/results/llmservingsim_sharegpt_300_hicache/aggregate_summary.md`：

| policy | pass | completed | prefix cache reused | disk prefetch | mean TTFT ms | mean TPOT ms | mean E2E ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | cold | 300 | 4.72% | 0.00% | 64872.36 | 29.24 | 83994.61 |
| lru | warm | 300 | 19.74% | 28.93% | 65534.61 | 34.67 | 88155.49 |
| lfu | cold | 300 | 4.74% | 0.00% | 64925.02 | 29.25 | 84053.07 |
| lfu | warm | 300 | 15.27% | 29.25% | 64176.10 | 34.05 | 86401.07 |

解读：

- ShareGPT 的自然会话 prefix 重叠较低，cold pass 只有约 4.7% prefix reuse 是合理结果。
- warm pass 的 disk prefetch 到约 29%，说明 storage write-through 和下一轮 prefetch 链路有效。
- LRU warm 的 prefix reuse 更高，但 LFU warm 的 TTFT/E2E 略好；策略优劣不能只看 prefix reuse。

### 8.2 SWE-bench 结论

汇总见 `hisim/test/results/llmservingsim_swebench_hicache/aggregate_summary.md`：

| scale | pass | completed | prefix reuse | disk prefetch | mean TTFT ms | mean TPOT ms | mean E2E ms | L2 load sum s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | cold | 40 | 91.37% | 0.00% | 68.78 | 22.73 | 5330.51 | 0.00 |
| 40 | warm | 40 | 94.89% | 7.03% | 50.15 | 22.16 | 5187.42 | 0.03 |
| 300 | cold | 300 | 89.89% | 0.00% | 126007.19 | 56.15 | 133720.05 | 143.84 |
| 300 | warm | 300 | 62.49% | 3.52% | 92448.52 | 34.41 | 97537.25 | 21.16 |

解读：

- SWE-bench agentic trace 有强上下文继承，prefix reuse 接近 90% 是 workload 特性，不应外推到普通会话。
- 300 条 warm pass 的 prefix reuse 下降但延迟改善，说明 host/storage 迁移改变了 radix cache 保留结构；应同时看 `disk_prefetch_ratio`、`L2 load sum` 和 E2E latency。
- 对 HiCache 策略评估，推荐至少同时报告 prefix reuse、disk prefetch、L2 load/backup、TTFT/TPOT/E2E。

### 8.3 对已有报告的优化建议

已有报告已经把实验命令、输入负载、关键指标写清楚。建议后续统一补三类信息：

1. 报告开头固定写明：工作区 commit/branch、SGLang version、HiSim config、是否启用 KVCM optimizer backend。
2. 每份报告都明确区分：prefix cache reuse、disk prefetch ratio、host/device L2 load/back 的含义。
3. 每份报告都加一个“不能外推的边界”：例如 SWE-bench 高 reuse 不能代表 ShareGPT，自然会话低 prefix overlap 不能否定 HiCache 在 agentic workload 的价值。

## 9. HiSim 的创新点

1. **框架控制流保真，而不是离线公式回放**  
   HiSim 复用 SGLang scheduler、radix cache、HiRadixCache 和请求队列，只替换模型执行与 cache/storage 操作。这比单纯用 CSV 公式估算更接近实际 serving 行为。

2. **低成本 CPU mock simulation**  
   真实模型 forward 被替换为空 logits 和固定采样，KV pool 变成 mock pool，因此可以在无 GPU 或低资源环境中跑 serving 级实验。

3. **batch shape 到硬件性能模型的可插拔抽象**  
   `ScheduleBatch(FakeRequest)` 把框架内部 batch 简化成 prefill/decode 预测所需的最小输入。这个抽象未来可以复用到 vLLM、TensorRT-LLM 等后端。

4. **trace replay 与 serving 指标闭环**  
   `bench_serving.py` 发送真实 token ids 和 timestamp，server 侧输出 TTFT、TPOT、ITL、E2E、throughput、prefix reuse、disk prefetch，能直接用于策略对比。

5. **多级 KVCache 行为可控实验**  
   通过 `max_total_tokens`、radix eviction policy、HiCache storage backend、write/prefetch policy，可以构造 cold/warm、LRU/LFU、不同容量压力的实验矩阵。

6. **模型/硬件事实与仿真场景分离**  
   `src/hisim/spec` 管模型和硬件事实，`configs/simulation` 管场景选择。README 中也强调 runtime HF config 优先、内置 registry 是 fallback。

## 10. 不足与风险

| 问题 | 影响 | 证据/位置 |
| --- | --- | --- |
| SGLang 内部 API 版本敏感 | class hook 绑定内部类名与方法，SGLang 升级容易失效。 | `launch_server.py:31-43`，`version.py:11-20`。 |
| README 支持矩阵与代码兼容版本不完全一致 | README 仍强调 `0.5.6.post2`，代码列到 `0.5.9`，需要统一稳定口径。 | `hisim/README.md` 与 `version.py:11-20`。 |
| vLLM 仿真未实现 | `bench_serving` 能向 vLLM 发 OpenAI 请求，但 server hook 和 config parser 只支持 SGLang。 | `bench_serving.py:875-887`，`config.py:163-178`。 |
| 多 instance 隔离未建模 | KVCM optimizer mock storage 写死单 `instance_id`，file backend 依赖路径隔离。 | `sglang_mock_class.py:1071-1072`。 |
| Predictor 粒度较粗 | prefill/decode 使用 batch 均值和修正系数，不能完全表达每个请求的序列分布、内核并发和调度细节。 | `aiconfigurator.py:447-518`。 |
| HiCache storage 只是行为模拟 | file backend 只存 key；带宽模型没有真实 storage queue、I/O contention、失败重试。 | `sglang_mock_class.py:1024-1164`。 |
| overlap schedule 被简化 | 初始化时禁用 SGLang overlap schedule，仿真和真实高并发运行仍有差距。 | `sglang_hook.py:599-616`。 |
| 部分测试依赖外部环境 | SGLang runner 测试会在缺少 SGLang/vLLM 时 skip，CI 不能保证所有路径都常跑。 | `hisim/test/test_simulation_sglang_runner.py:19-25`。 |

## 11. 待优化方向

1. **抽象 backend adapter**  
   把当前 SGLang 专属 hook 拆成公共层和后端层：`BackendAdapter` 负责请求元数据、batch shape、cache hit、load/save event、profile flush。

2. **补齐 Instance 隔离**  
   `instance_id` 应从配置、请求或 server args 进入 HiSim storage；storage key 文件也应按 instance namespace 隔离。需要新增跨 instance 不命中的回归测试。

3. **统一指标 schema**  
   固定 `metrics.json`、`request.jsonl`、`iteration.jsonl` 字段定义和单位，明确哪些字段来自 SGLang、哪些字段来自 HiSim。

4. **增强 HiCache 模型**  
   引入 host capacity、storage capacity、I/O queue、并发 prefetch、失败/timeout、网络 storage 等模型，不只按带宽线性估算。

5. **完善 predictor 校准**  
   为每个模型/硬件/backend 维护校准配置、版本、误差报告；将 XGBoost decode correction 的启用状态写入 metrics。

6. **版本兼容测试**  
   对 `0.5.6.post2`、`0.5.7`、`0.5.8`、`0.5.9` 分别跑最小 hook import 和 smoke test，避免兼容列表漂移。

7. **文档分层**  
   `README` 保持快速上手；`test/reports` 保留实验记录；新增 `docs` 或本报告这种 architecture guide 讲清设计边界。

## 12. vLLM 接入评估

### 12.1 当前仓库已有的 vLLM 能力

项目根 README 明确说 KVCache Manager Client/Connector 支持 vLLM、SGLang、RTP-LLM、TRT-LLM。源码中也已有：

```text
kv_cache_manager/py_connector/vllm/
```

其中 `v1_connector.py` 继承 vLLM `KVConnectorBase_V1`，读取 `VllmConfig`、`SchedulerOutput`、`KVConnectorOutput` 等 vLLM V1 对象，并通过 `KvCacheManagerClient` 注册 instance、查询 cache location、执行 load/save。关键位置：

- vLLM connector 基类和类型导入：`kv_cache_manager/py_connector/vllm/v1_connector.py:13-25`。
- connector extra config 需要 `manager_uri`、`instance_group`、`instance_id` 等字段：`kv_cache_manager/py_connector/vllm/config.py:1-40`。
- 初始化时向 Manager 注册 instance 和 location spec：`kv_cache_manager/py_connector/vllm/v1_connector.py:123-194`。
- location query 使用 `instance_id` 做 prefix match：`kv_cache_manager/py_connector/vllm/location_query_manager.py:65-105`。

这说明 **Tair KVCache Manager 对接 vLLM 是已有方向**。

### 12.2 当前 HiSim 为什么还不能直接接 vLLM

HiSim 的仿真 server 入口是 `hisim.simulation.sglang.launch_server`，启动前安装的全是 SGLang hook，见 `launch_server.py:12-25`。`ConfigManager._parse_server_args()` 也只认识 `backend == "sglang"`，否则直接 `RuntimeError`，见 `config.py:163-178`。

`bench_serving.py` 虽然支持 `--backend vllm` 和 `vllm-chat`，见 `bench_serving.py:875-887`，但这只是 client 侧向 OpenAI-compatible API 发请求；它不会让 vLLM server 自动使用 HiSim 的 mock model、global clock、prefix/cache 指标输出。

所以当前状态是：

- **Manager connector 可接 vLLM**：已有 `py_connector/vllm`。
- **HiSim 仿真不可直接接 vLLM**：缺少 `hisim.simulation.vllm` 的 server/runner/hook。

### 12.3 可行接入方案

vLLM 官方 stable docs 中，`KVConnectorBase_V1` 提供 scheduler 侧和 worker 侧 KV transfer primitives：scheduler 侧包括 `get_num_new_matched_tokens()`、`update_state_after_alloc()`、`request_finished()`、`take_events()`；worker 侧包括 `start_load_kv()`、`wait_for_layer_load()`、`save_kv_layer()`、`wait_for_save()`。官方文档也说明 `register_kv_caches()` 可用于把各层 KV cache 预注册到 connector。参考：

- vLLM stable `KVConnectorBase_V1` API: <https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/base/>
- vLLM stable architecture overview: <https://docs.vllm.ai/en/stable/design/arch_overview/>

基于这些接口，HiSim 接 vLLM 有两条路线：

| 路线 | 做法 | 优点 | 代价 |
| --- | --- | --- | --- |
| A. Hook vLLM scheduler/model runner/cache manager | 类似 SGLang，拦截 vLLM V1 scheduler、model runner、KV cache manager，替换真实 forward 并生成 HiSim metrics。 | 最大程度复用现有 HiSim 模式，指标链路一致。 | vLLM 内部 API 变化快，维护成本高。 |
| B. 基于 vLLM KVConnector V1 做仿真 connector | 用 connector 接管 remote KV match/load/save，在 vLLM 中保留部分真实调度，再通过 connector 事件和 scheduler output 构造 HiSim metrics。 | 更贴近 vLLM 官方扩展点，也能复用现有 `py_connector/vllm` 经验。 | 仍需解决模型 forward mock、global clock、request latency 统计，不能只靠 connector 完成完整仿真。 |

推荐路线是 **B + 少量必要 hook**：

1. 新增 `hisim/src/hisim/simulation/vllm/`。
2. 复用 `time_predictor.ScheduleBatch`，新增 vLLM batch adapter，将 vLLM scheduled requests/block tables 转成 `FakeRequest`。
3. 用 vLLM KVConnector V1 或其相邻扩展点获取 remote matched tokens、load/save events、request finish events。
4. 对 vLLM model runner 做最小 mock，避免真实权重 forward。
5. 新增 `ConfigManager._parse_server_args(..., backend="vllm")`。
6. 输出同一套 `metrics.json/request.jsonl/iteration.jsonl`。
7. 对 `instance_id` 做强制配置，保证跨 instance 不匹配。

### 12.4 最小里程碑

| 阶段 | 目标 | 验收标准 |
| --- | --- | --- |
| M1 | vLLM backend skeleton | `python -m hisim.simulation.vllm.launch_server --sim-config-path ...` 能启动并拒绝未支持参数。 |
| M2 | request replay 元数据 | Hisim collection 请求能携带 `created_time/total_request` 进入 vLLM scheduler。 |
| M3 | model forward mock | 不加载真实权重即可完成固定 token 输出。 |
| M4 | batch latency prediction | vLLM scheduled batch 能转换为 `ScheduleBatch` 并推进 `StateManager`。 |
| M5 | prefix/cache metrics | 输出 `final_reused_tokens`、`prefetch_complete_tokens`、TTFT/TPOT/E2E。 |
| M6 | KVCM/vLLM connector 联动 | 用 `instance_id` 隔离的 Manager connector 或 mock connector 做 cold/warm replay。 |

## 13. 结论

HiSim 当前最有价值的部分不是某个单独公式，而是“真实 serving 控制流 + 可替换性能模型 + trace replay + KVCache 指标”的闭环。现有 SGLang 路径已经能支撑 ShareGPT/SWE-bench 的 HiCache 策略实验。

它的主要短板也清晰：SGLang hook 版本敏感、多 instance 隔离未充分建模、HiCache storage 和 predictor 仍是近似模型、vLLM 仿真后端尚未实现。后续如果目标是“基于 vLLM 的 KVCache 仿真”，建议不要绕过现有 vLLM connector，而是在 HiSim 中新增 vLLM backend adapter，并把 `ScheduleBatch`、`StateManager`、metrics schema 作为公共核心复用。
