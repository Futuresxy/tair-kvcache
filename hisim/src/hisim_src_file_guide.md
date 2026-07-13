# HiSim `src/hisim` 源码导览

本文档面向初次学习 HiSim 源码的读者，说明
`hisim/src/hisim` 下每个目录和 Python 文件的职责、作用、关键对象和阅读重点。

分析范围：

- 源码根目录：`hisim/src/hisim`
- 文件数量：45 个 Python 文件
- 不包含：`__pycache__`、测试文件、下载数据、报告文件和外部依赖包源码

## 1. 总体定位

HiSim 是一个面向 LLM 推理服务的仿真工具。它通过 hook 改写 SGLang 的关键类，把真实模型 forward、真实 GPU kernel 和真实 KV cache IO 替换成轻量 mock 行为，同时用 `time_predictor` 预测每个 batch 的推理耗时，再用仿真时钟统计 TTFT、TPOT、吞吐、prefix cache 命中、L2/L3 KV cache IO 等指标。

从 `src/hisim` 的代码分层看，核心链路是：

```text
dataset/ 生成或加载请求
  -> simulation/bench_serving.py 作为压测客户端发请求
  -> simulation/sglang/launch_server.py 启动被 hook 的 SGLang 服务
  -> hook/ 在 import/class 创建阶段替换 SGLang 类
  -> simulation/sglang/sglang_hook.py 接管调度、KV cache、指标
  -> time_predictor/ 用 AIConfigurator 预测 batch latency
  -> simulation/manager/ 维护配置、环境变量、全局仿真时钟
  -> simulation/utils.py 汇总指标
```

## 2. 目录职责总览

| 目录 | 职责 | 学习重点 |
| --- | --- | --- |
| `hisim/` | Python 包入口，声明版本。 | 只包含 `__version__`。 |
| `hisim/dataset/` | 把随机负载、ShareGPT、HiSim trace 转成统一请求对象。 | `GenericRequest`、`DatasetArgs`、`get_dataset()`。 |
| `hisim/hook/` | 通用 hook 框架，支持 class hook 和 module hook。 | 如何在 SGLang 类创建时替换方法。 |
| `hisim/simulation/` | 仿真客户端、仿真配置、指标、SGLang 接入层。 | 主流程所在目录。 |
| `hisim/simulation/base/` | benchmark runner 抽象接口。 | `BaseBenchmarkRunner`。 |
| `hisim/simulation/manager/` | 全局配置、环境变量、仿真时钟和状态管理。 | `ConfigManager`、`StateManager`、`Envs`。 |
| `hisim/simulation/sglang/` | SGLang 专用 hook、mock KV cache、服务启动和 benchmark runner。 | HiSim 与 SGLang 接入的核心。 |
| `hisim/spec/` | 模型、硬件、数据类型规格定义与注册表。 | `ModelInfo`、`AcceleratorInfo`、`DataType`。 |
| `hisim/spec/accelerator/` | 硬件规格库和硬件信息对象。 | H20/H100/A100/Ascend 等设备注册。 |
| `hisim/spec/model/` | 模型规格库和模型信息对象。 | Qwen、DeepSeek 内置模型注册。 |
| `hisim/time_predictor/` | 推理时间预测器接口和 AIConfigurator 适配器。 | `ScheduleBatch`、`AIConfiguratorTimePredictor`。 |
| `hisim/utils/` | 日志和 JSON 序列化工具。 | `get_logger()`、`CustomJsonEncoder`。 |

## 3. 核心运行流程

### 3.1 服务端仿真启动

入口是 `hisim/simulation/sglang/launch_server.py`：

1. 如果没有 CUDA，则安装 `sgl_kernel.load_utils` 的 module hook，避免 CPU 环境导入 SGLang kernel 失败。
2. 安装 SGLang class hook：`Scheduler`、`ModelRunner`、`TokenizerManager`、`StorageBackendFactory`、`HiCacheController`、`HiRadixCache`。
3. 解析原生 SGLang 参数和 `--sim-*` 仿真参数。
4. 将仿真配置写入或读取 `HISIM_CONFIG_PATH`。
5. 调用 SGLang 原生 `launch_server(server_args)`。

### 3.2 客户端压测与请求注入

入口是 `hisim/simulation/bench_serving.py`：

1. 按数据集参数生成 `DatasetRow`。
2. 按 `request_rate` 或 trace timestamp 产生请求。
3. 对 SGLang `/generate` 请求，在 `sampling_params.custom_params.simulation` 中写入 `created_time`、`total_request` 等仿真信息。
4. `bench_mode=simulation` 时不按真实时间 sleep，而是让服务端 hook 通过全局仿真时钟推进时间。
5. 仿真结束后读取服务端写出的 `/tmp/hisim/output/metrics.json`。

### 3.3 SGLang 调度 hook 与时间推进

核心在 `hisim/simulation/sglang/sglang_hook.py` 的 `C_SchedulerHook`：

1. `wrapped_init()` 初始化模型信息、硬件信息、调度配置和 `AIConfiguratorTimePredictor`。
2. `wrapped_recv_requests()` 在 OFFLINE 模式下先收齐所有请求，再按 timestamp 放入 future queue。
3. `wrapped_run_batch()` 把 SGLang batch 转成 HiSim 的 `ScheduleBatch`，调用预测器得到 batch latency。
4. `wrapped_process_batch_result()` 根据预测 latency、L2 load/backup 耗时推进 `StateManager` 的全局仿真时钟，并记录每个请求的 token latency。
5. `wrapped_profile()` 输出 `metrics.json`、`iteration.jsonl`、`request.jsonl`。

### 3.4 KV cache 与 HiCache 模拟

核心在 `hisim/simulation/sglang/sglang_mock_class.py` 和 `sglang_hook.py`：

- `MockReqToTokenPool` 模拟 request 到 token slot 的映射。
- `MockTokenToKVPool` 模拟 device 端 KV cache pool，但实际 head/layer 维度被简化以节省内存。
- `MockPagedTokenToKVPoolAllocator` 模拟 paged KV cache 分配。
- `MockTokenToKVPoolHost` 模拟 host memory KV cache，并根据内存带宽估算 H2D/D2H 时间。
- `MockHiCacheStorage` 模拟外部 storage backend；如果 `kv_cache_manager.optimizer.pybind` 可用，则接入 KVCM，否则使用本地 set/file 模拟 key 存储。

注意：`MockHiCacheStorage` 里当前 KVCM 路径使用单个固定 `instance_id`，而项目约束要求 KVCache 只能在同一 `instance_id` 内复用，跨 instance 不匹配。后续如做多实例仿真，需要补齐 instance 隔离。

## 4. 文件职责列表

### 4.1 包根目录

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `hisim/__init__.py` | 包版本声明。 | `__version__ = "0.1.0"` |

### 4.2 `dataset/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `dataset/__init__.py` | 数据集注册表和统一工厂入口。根据 `DatasetArgs.name` 创建具体数据集；如果设置 `prefix_hit_rate`，再包一层 `PrefixCacheDecorator`。 | `dataset_registry`、`get_dataset()` |
| `dataset/base_dataset.py` | 数据集抽象和统一请求结构。所有数据集最终都返回 `GenericRequest`。 | `GenericRequest`、`BaseDataset`、`SimpleDataset` |
| `dataset/dataset_args.py` | 数据集配置结构，描述请求数、输入/输出长度范围、数据文件路径、prefix 命中率。 | `DatasetArgs` |
| `dataset/random.py` | 随机请求数据集。可生成随机 token ids、随机文本 prompt、所有请求相同 token 的特殊数据集。 | `RandomIDsDataset`、`RandomDataset`、`IdenticalIDsDataset` |
| `dataset/share_gpt.py` | 从 ShareGPT JSON 文件采样 prompt，并把 token 长度扩展/截断到指定范围。 | `ShareGPTDataset` |
| `dataset/hisim_collection.py` | 加载 HiSim 自定义 JSONL trace，每行包含 `input_ids`、`output_length`、`timestamp/created_time`，并对 timestamp 做归零对齐。 | `HisimCollectionDataset` |
| `dataset/prefix_cache.py` | 数据集装饰器。按 `prefix_hit_rate` 让后续请求复用前一个请求的一段 prefix，用于模拟 prefix cache 命中。 | `PrefixCacheDecorator` |

### 4.3 `hook/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `hook/__init__.py` | 导出 hook 安装/卸载 API。 | `install_class_hooks`、`install_module_hooks` |
| `hook/base_hook.py` | hook 基类和注册辅助函数。每个 hook 通过 `HOOK_CLASS_NAME`、`HOOK_MODULE_NAME` 指明目标。 | `BaseHook`、`_register_hooks()` |
| `hook/class_hook_entry.py` | class hook 机制。通过替换 `builtins.__build_class__`，在目标类创建时调用 hook 修改或替换类。 | `install_class_hooks()`、`remove_class_hooks()` |
| `hook/module_hook_entry.py` | module hook 机制。通过 `sys.meta_path` 自定义 finder/loader，在目标模块导入时执行原模块代码后调用 hook。 | `HookMetaPathFinder`、`HookLoader`、`install_module_hooks()` |
| `hook/utils.py` | 从函数参数中按完整类型名查找对象，供 hook 包装函数解析 SGLang 参数对象。 | `get_obj_from_args()` |

### 4.4 `simulation/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `simulation/__init__.py` | 空包标记文件。 | 无 |
| `simulation/bench_serving.py` | 在线 serving benchmark 客户端，改造自 vLLM/SGLang benchmark。支持 normal/simulation 模式、多后端请求、数据集采样、trace replay、metrics 计算和结果落盘。 | `RequestFuncInput`、`RequestFuncOutput`、`DatasetRow`、`BenchmarkMetrics`、`run_benchmark()`、`benchmark()` |
| `simulation/sim_args.py` | 仿真配置 dataclass 和 CLI 参数解析。用于 `launch_server.py` 将 `--sim-*` 参数转换为 JSON 配置。 | `SimulationArgs`、`PlatformConfig`、`PredictorConfig`、`SchedulerConfig` |
| `simulation/types.py` | 仿真运行期通用数据结构。区分 benchmark、scheduler、platform、request stats 和 mock simulation mode。 | `BenchmarkConfig`、`SchedulerConfig`、`RequestStats`、`PlatformConfig`、`MockSimulationMode` |
| `simulation/utils.py` | KV cache 容量估算和指标汇总。根据模型/并行度计算每 token KV cache 元素数，并统计 TTFT/TPOT/ITL/吞吐等指标。 | `calc_kv_cache_cell_elems()`、`estimate_kv_cache_pool_capacity()`、`calc_metrics()` |

`simulation/bench_serving.py` 体量较大，建议按功能分段理解：

| 代码区域 | 作用 |
| --- | --- |
| 顶部 dataclass 和 `async_request_*` | 不同后端 HTTP 请求实现：SGLang `/generate`、OpenAI completions/chat、TensorRT-LLM、Truss 等。 |
| `get_dataset()` 和 `sample_*_requests()` | 构造 ShareGPT、random、image、MMMU、Mooncake、HiSim collection 等输入请求。 |
| `get_request()` | 按泊松到达或 trace timestamp 产生请求，并写入 simulation 参数。 |
| `calculate_metrics()` | 根据客户端观测的输出统计 benchmark 指标。 |
| `load_simulation_metrics()` | simulation 模式下改用服务端 hook 输出的真实仿真指标。 |
| `benchmark()` | warmup、profile start/stop、并发请求调度、结果汇总的主协程。 |
| `run_benchmark()` | CLI/Namespace 入口，设置 URL、模型、tokenizer、数据集和参数默认值。 |

### 4.5 `simulation/base/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `simulation/base/__init__.py` | 空包标记文件。 | 无 |
| `simulation/base/runner.py` | benchmark runner 抽象接口，定义 benchmark、flush_cache、shutdown 三个能力。 | `BaseBenchmarkRunner` |

### 4.6 `simulation/manager/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `simulation/manager/__init__.py` | 统一导出 manager。 | `StateManager`、`Envs`、`ConfigManager` |
| `simulation/manager/env.py` | 读取仿真环境变量。包括配置路径、输出目录、仿真模式、warmup 数量、HiCache storage 路径和是否重置 storage。 | `Envs` |
| `simulation/manager/state.py` | 全局仿真状态。维护 iteration、global clock、当前/上次 inference latency、L2 load/backup 累计时间。 | `StateManager` |
| `simulation/manager/config.py` | 从 `HISIM_CONFIG_PATH` 和 SGLang server args 组合出模型、硬件、调度配置，并创建预测器。 | `ConfigManager` |

`ConfigManager` 是连接规格、配置和预测器的关键点：

- `get_model_info()` 从 HuggingFace config 或配置文件得到 `ModelInfo`。
- `get_accelerator_info()` 从 `platform.accelerator.name` 查硬件注册表。
- `get_scheduler_config()` 合并 SGLang server args 和 `scheduler` 配置。
- `get_inference_time_predictor()` 根据 `predictor.name` 创建 `AIConfiguratorTimePredictor`。

### 4.7 `simulation/sglang/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `simulation/sglang/__init__.py` | 空包标记文件。 | 无 |
| `simulation/sglang/launch_server.py` | HiSim 仿真服务端启动入口。先安装 hook，再调用 SGLang 原生 `launch_server`。 | `SimulationArgs`、`install_class_hooks()` |
| `simulation/sglang/sgl_kernel_hook.py` | CPU 环境下绕过 `sgl_kernel.load_utils` 的 architecture-specific ops 加载错误。 | `M_SGLangKernelLoadUtilHook` |
| `simulation/sglang/sglang_bench.py` | 进程内 SGLang benchmark runner。直接创建 SGLang `Engine`，发送 dataset 请求，并读取 hook 输出的 metrics/iteration/request 文件。 | `SGLangBenchmarkRunner` |
| `simulation/sglang/sglang_hook.py` | SGLang class hook 主文件。替换 Engine、TokenizerManager、ModelRunner、HiCacheController、HiRadixCache、StorageBackendFactory、Scheduler 的关键方法。 | `C_ModelRunnerHook`、`C_SchedulerHook`、`C_HiCacheController`、`C_HiRadixCacheHook` |
| `simulation/sglang/sglang_mock_class.py` | SGLang KV cache、allocator、host cache、storage backend 的 mock 实现。避免真实模型权重/真实 KV cache 大量占用，同时保留调度和缓存行为。 | `MockReqToTokenPool`、`MockTokenToKVPool`、`MockPagedTokenToKVPoolAllocator`、`MockTokenToKVPoolHost`、`MockHiCacheStorage` |
| `simulation/sglang/version.py` | SGLang 版本兼容调度器。针对不同 SGLang 版本选择不同 wrapper 实现。 | `COMPATIBLE_VERSIONS`、`VersionDispatcher` |

`sglang_hook.py` 的主要 hook 说明：

| Hook 类 | 目标 SGLang 类/模块 | 作用 |
| --- | --- | --- |
| `C_EngineHook` | `sglang.srt.entrypoints.engine.Engine` | 给 Engine 增加 `clear_hicache_storage()`。当前在启动列表中没有安装此 hook。 |
| `C_TokenizerManagerHook` | `TokenizerManager` | 在 blocking 模式下把服务端收到请求的时间写入 simulation 参数。 |
| `C_ModelRunnerHook` | `ModelRunner` | 替换模型初始化、forward、sample。创建 mock model、mock KV pool，forward 只返回空 logits，不执行真实模型。 |
| `C_HiCacheController` | `HiCacheController` | 禁用原异步 prefetch/backup thread，改为由仿真主循环按当前 inference 时间片推进 prefetch/backup。 |
| `C_HiRadixCacheHook` | `HiRadixCache` | 替换 host KV cache pool，接入 mock storage backend，并在检查事件时处理 backup/prefetch。 |
| `C_StorageBackendFactory` | `StorageBackendFactory` | 强制创建 `MockHiCacheStorage`。 |
| `C_SchedulerHook` | `Scheduler` | HiSim 的核心调度 hook。收请求、构造 `ScheduleBatch`、调用预测器、推进全局时钟、输出 metrics。 |

`sglang_mock_class.py` 的主要 mock 对象说明：

| 对象 | 作用 |
| --- | --- |
| `alloc_extend_cpu()` | CPU 版 paged extend 分配逻辑，替代部分 GPU kernel 行为。 |
| `alloc_decode_cpu()` | CPU 版 decode 分配逻辑。 |
| `get_num_new_pages()` | 计算 extend/decode 需要的新 page 数。 |
| `MockReqToTokenPool` | 请求到 token slot 的映射池，兼容 SGLang 0.5.6 到 0.5.9 的 alloc/free 差异。 |
| `MockTokenToKVPool` | device KV pool，保留接口但将实际 KV 维度压缩为 `head_num=1, head_dim=1`，减少内存占用。 |
| `MockBaseTokenToKVPoolAllocator` | KV allocator 基类，提供 state backup/restore、free group、可用容量接口。 |
| `MockTokenToKVPoolAllocator` | 非 paged KV allocator。 |
| `MockPagedTokenToKVPoolAllocator` | paged KV allocator，支持 `alloc_extend()`、`alloc_decode()`。 |
| `MockTokenToKVPoolHost` | host KV cache pool，估算 host-device load/backup 带宽时间，并累计到 `StateManager`。 |
| `MockHiCacheStorage` | L3/storage backend mock。可走 KVCM 优化器，也可用本地 set/file 存 key。 |

### 4.8 `spec/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `spec/__init__.py` | 统一导出规格对象。 | `AcceleratorInfo`、`ModelInfo`、`DataType` |
| `spec/data_type.py` | 数据类型枚举、字节数映射、别名和 torch dtype 到 HiSim dtype 的转换。 | `DataType` |

### 4.9 `spec/accelerator/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `spec/accelerator/__init__.py` | 导出硬件规格对象和内置硬件集合。 | `AcceleratorInfo`、`NVIDIA`、`Huawei` |
| `spec/accelerator/base.py` | 硬件信息 dataclass 和硬件注册表。支持按别名查询、FLOPS/HBM/带宽单位转换。 | `AcceleratorInfo` |
| `spec/accelerator/info.py` | 内置硬件规格注册。包括 A100、H100、H200、H20、RTX4090、Ascend 910B、Ascend 950。 | `NVIDIA`、`Huawei` |

### 4.10 `spec/model/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `spec/model/__init__.py` | 导出模型规格对象和内置模型集合。 | `ModelInfo`、`Qwen`、`DeepSeek` |
| `spec/model/base.py` | 模型信息 dataclass、模型注册表、HuggingFace/ModelScope config 解析。负责补齐 `head_dim`、`kv_hidden_dim`、`max_seq_len` 等派生字段。 | `ModelInfo`、`model_config_mapping_dict` |
| `spec/model/info.py` | 内置模型规格注册。包括 Qwen2.5、Qwen3 dense/MoE、DeepSeek V3/R1 等。 | `Qwen`、`DeepSeek` |

`ModelInfo.__post_init__()` 里有一个重要约束：`num_hidden_layers` 必须等于 full attention、linear attention、sliding attention 的层数之和。普通模型在 `from_dict()` 中默认将全部层计入 `num_full_attention`。

### 4.11 `time_predictor/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `time_predictor/__init__.py` | 导出预测器接口和 AIConfigurator 实现。 | `InferTimePredictor`、`AIConfiguratorTimePredictor` |
| `time_predictor/base.py` | 推理时间预测器抽象接口和 HiSim 内部 batch/request 表示。 | `FakeRequest`、`ScheduleBatch`、`InferTimePredictor` |
| `time_predictor/aiconfigurator.py` | AIConfigurator 适配器。把 HiSim 模型/硬件/调度配置转换成 AIConfigurator `ModelConfig`/`RuntimeConfig`，查实测数据库预测 prefill/decode latency，并可用 XGBoost 修正 decode attention。 | `AIConfiguratorTimePredictor`、`get_perf_model()` |

`AIConfiguratorTimePredictor.predict_infer_time()` 是仿真延时预测核心：

- decode：用 batch 内 `past_kv_length` 的平均值作为 `isl`，`osl=2`；如果有 XGBoost bucket model，则根据每个请求的 `past_kv_length` 分布预测 `aic_gen_attn_ms / measured_gen_attn_ms`，再取倒数作为 generation attention 修正比例。
- prefill：用平均 `past_kv_length + input_length` 作为 `isl`，平均 `past_kv_length` 作为 `prefix`，并用实际 FLOPs 与平均 FLOPs 的比值修正 sequence imbalance。
- 最后调用 AIConfigurator `InferenceSession.run_static()`，取 latency dict 求和，单位从 ms 转成秒。

### 4.12 `utils/`

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `utils/__init__.py` | 导出日志工具。 | `get_logger` |
| `utils/logger.py` | 创建 HiSim logger，设置统一 formatter，避免重复添加 handler。 | `get_logger()` |
| `utils/json.py` | JSON 编码器，支持 Enum、dataclass、numpy scalar、numpy array。用于 metrics 和对象序列化。 | `CustomJsonEncoder` |

## 5. 关键数据结构

| 数据/对象 | 定义位置 | 作用 |
| --- | --- | --- |
| `GenericRequest` | `dataset/base_dataset.py` | 数据集层统一请求，包含 prompt/token_ids、输入长度、输出长度、自定义参数。 |
| `DatasetArgs` | `dataset/dataset_args.py` | 数据集构造参数。 |
| `DatasetRow` | `simulation/bench_serving.py` | benchmark 客户端内部请求格式，支持文本、多模态、timestamp、simulation dict。 |
| `RequestFuncInput` | `simulation/bench_serving.py` | HTTP 请求函数输入。 |
| `RequestFuncOutput` | `simulation/bench_serving.py` | HTTP 请求函数输出，记录 latency、TTFT、ITL、输出长度等。 |
| `RequestStats` | `simulation/types.py` | 服务端 hook 侧的请求统计结构。 |
| `FakeRequest` | `time_predictor/base.py` | 预测器输入请求，只保留 `input_length` 和 `past_kv_length`。 |
| `ScheduleBatch` | `time_predictor/base.py` | 预测器输入 batch，可判断 prefill/decode。 |
| `SchedulerConfig` | `simulation/types.py` | 预测器和调度侧的并行度、dtype、backend、page size 等配置。 |
| `PlatformConfig` | `simulation/types.py` | 硬件和 L2/L3 cache 带宽配置。 |
| `ModelInfo` | `spec/model/base.py` | 模型结构规格。 |
| `AcceleratorInfo` | `spec/accelerator/base.py` | 硬件规格。 |

## 6. 主要依赖关系

```text
simulation/sglang/launch_server.py
  -> hook/
  -> simulation/sim_args.py
  -> simulation/sglang/sglang_hook.py

simulation/sglang/sglang_hook.py
  -> simulation/manager/
  -> time_predictor/
  -> simulation/sglang/sglang_mock_class.py
  -> simulation/utils.py

time_predictor/aiconfigurator.py
  -> spec/
  -> aiconfigurator.sdk
  -> xgboost

simulation/manager/config.py
  -> spec/
  -> simulation/types.py
  -> time_predictor/

dataset/
  -> simulation/bench_serving.py
  -> simulation/sglang/sglang_bench.py
```

## 7. 配置与环境变量

| 配置/环境变量 | 读取位置 | 作用 |
| --- | --- | --- |
| `HISIM_CONFIG_PATH` | `simulation/manager/env.py` | 仿真 JSON 配置路径。 |
| `HISIM_OUTPUT_DIR` | `simulation/manager/env.py` | metrics/iteration/request 输出目录，默认 `/tmp/hisim/output/`。 |
| `HISIM_SIMULATION_MODE` | `simulation/manager/env.py` | `OFFLINE` 或 `BLOCKING`。OFFLINE 模式按仿真时钟回放。 |
| `HISIM_NUM_WARMUP` | `simulation/manager/env.py` | 输出 metrics 时跳过的 warmup 请求数量。 |
| `HISIM_RESET_HICACHE_STORAGE` | `simulation/manager/env.py` | 是否清空 mock storage cache。 |
| `HISIM_HICACHE_STORAGE_PATH` | `simulation/manager/env.py` | 本地 set/file storage 的 key 文件路径。 |
| `--sim-*` CLI 参数 | `simulation/sim_args.py` | 启动服务时生成或覆盖仿真配置。 |

配置文件中的关键段落：

| 配置段 | 消费位置 | 作用 |
| --- | --- | --- |
| `platform` | `ConfigManager.get_platform_config()` | 硬件、磁盘带宽、内存带宽、节点设备数。 |
| `predictor` | `ConfigManager.get_inference_time_predictor()` | 预测器类型、AIConfigurator 数据库路径、device name、scale factor、XGBoost 路径。 |
| `scheduler` | `ConfigManager.get_scheduler_config()` | TP/EP/DP/PP、dtype、KV dtype、backend version。 |
| `model` | `ConfigManager.get_model_info()` | 当没有 HF config 时，按内置模型注册表查模型。 |

## 8. 建议阅读顺序

1. `hisim/README.md`：先跑通启动服务和 benchmark 的基本命令。
2. `simulation/sglang/launch_server.py`：理解服务端为什么先安装 hook 再导入/启动 SGLang。
3. `hook/base_hook.py`、`hook/class_hook_entry.py`：理解 class hook 的基础机制。
4. `simulation/sglang/sglang_hook.py`：重点看 `C_SchedulerHook`、`C_ModelRunnerHook`、`C_HiCacheController`。
5. `time_predictor/base.py`、`time_predictor/aiconfigurator.py`：理解 batch 怎么转成预测器输入、如何预测延时。
6. `simulation/manager/config.py`、`simulation/manager/state.py`：理解配置和仿真时钟。
7. `simulation/sglang/sglang_mock_class.py`：理解 KV cache pool、host cache、storage mock。
8. `dataset/`：理解如何生成请求和 trace replay。
9. `simulation/bench_serving.py`：最后看客户端压测和结果汇总。

## 9. 修改入口建议

| 修改目标 | 建议位置 | 注意事项 |
| --- | --- | --- |
| 新增一种 workload/dataset | `dataset/` 和 `dataset/__init__.py` | 返回 `GenericRequest`；如果要支持 timestamp，写入 `custom_params.created_time`。 |
| 新增 benchmark 数据集类型 | `simulation/bench_serving.py:get_dataset()` | 返回 `DatasetRow`，并在 `get_request()` 中正确传递 simulation dict。 |
| 新增硬件 | `spec/accelerator/info.py` | 需要设置别名、HBM、带宽、不同 dtype TFLOPS。 |
| 新增模型 | `spec/model/info.py` 或 `ModelInfo.from_json()` | 确保 `num_hidden_layers`、attention layer 计数、KV heads、head_dim 正确。 |
| 更换预测器 | `time_predictor/base.py`、`simulation/manager/config.py` | 实现 `InferTimePredictor.predict_infer_time()` 并在 `ConfigManager` 中注册。 |
| 调整 AIConfigurator 预测 | `time_predictor/aiconfigurator.py` | 注意 dtype 映射、database path、XGBoost bucket 模型和 ms/s 单位转换。 |
| 修改 SGLang 调度仿真 | `simulation/sglang/sglang_hook.py:C_SchedulerHook` | 这是最敏感路径，涉及请求到达、batch latency、global clock、metrics。 |
| 修改 KV cache/HiCache 行为 | `simulation/sglang/sglang_mock_class.py`、`C_HiCacheController`、`C_HiRadixCacheHook` | 需要同时维护 device pool、host pool、storage 命中、L2/L3 IO 时间。 |
| 接入真实/外部 KV storage | `MockHiCacheStorage` | 必须满足 instance 隔离约束，避免跨 `instance_id` 复用 KVCache。 |

## 10. 风险和限制

- `simulation/sglang/sglang_hook.py` 强依赖 SGLang 内部类名和方法签名。SGLang 版本升级后，hook 可能失效，因此有 `VersionDispatcher` 但覆盖仍有限。
- `MockTokenToKVPool` 为降低内存占用将 `head_num/head_dim` 压缩为 1，适合调度和缓存仿真，不代表真实 KV tensor 形状。
- `MockHiCacheStorage` 在 KVCM 路径中使用固定 `instance_id`，当前不支持多实例隔离；这与项目级约束相关，是后续优化重点。
- AIConfigurator 数据库和 XGBoost 模型路径来自配置，路径错误会导致预测器初始化失败。
- `bench_serving.py` 支持多后端，但 HiSim 深度 hook 目前主要围绕 SGLang；vLLM/OpenAI 后端只作为 benchmark 客户端请求格式存在，没有同等深度的内部调度 hook。
- OFFLINE 模式会先收齐请求再按 trace timestamp 推进仿真时钟，适合离线回放；BLOCKING 模式会用 `time.sleep(predicted_latency)` 模拟真实等待。

## 11. 一句话记忆

HiSim 的 `src/hisim` 可以按五块记：

```text
spec 定义模型/硬件
dataset 准备请求
hook 改写 SGLang
time_predictor 预测 batch 延时
simulation 把请求、调度、KV cache、时钟和指标串起来
```
