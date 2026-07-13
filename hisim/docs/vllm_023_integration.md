# vLLM 0.23 接入设计与复现实验入口

## 接入层次

当前同时提供三种入口：

1. `VllmSchedulerSimulator`：直接驱动 vLLM 0.23.0 `Scheduler`、`KVCacheManager`、
   block allocator 和 prefix block hasher，用逻辑时钟快速回放大型 workload。
2. `create_hisim_llm()`：返回原生 `vllm.LLM` offline 对象，但通过 vLLM 官方
   `worker_cls` 扩展点加载 `HiSimWorker`。
3. `hisim.simulation.vllm.launch_server`：启动完整 vLLM OpenAI server、EngineCore、
   Scheduler、tokenizer、output processor 和 Prometheus metrics，只替换模型 Worker。

`HiSimWorker` 跳过 CUDA/NCCL 初始化、模型权重加载、KV tensor 分配、kernel compilation 和
GPU forward；它仍让原生 EngineCore 完成 KV block 规划和请求调度，然后根据
`SchedulerOutput` 调用 HiSim latency profile、产生合法 `ModelRunnerOutput`。因此这不是仿写的
HTTP server 或 scheduler，而是完整 vLLM 引擎中的模型执行替换。

这不是最终校准结果。默认 `uncalibrated-linear` 延迟参数仅用于验证调度闭环；只有
`calibrated: true` 且带有 RTX 4090 原始测量文件的 profile 才能用于仿真精度结论。

## Instance 隔离

一个 `VllmSchedulerSimulator` 对象只对应一个 `instance_id` 和一个 KVCache namespace。
请求的 `instance_id` 不匹配时立即报错。不同 Instance 必须创建不同 simulator，不能共享
同一个 vLLM scheduler 或 block pool。

## 输入与输出

workload 是 JSONL，每行一个 token 化后的请求：

```json
{"request_id":"r0","instance_id":"replica-0","prompt_token_ids":[1,2,3],"max_tokens":8,"arrival_time_ms":0}
```

结果 JSON 包含：

- 每请求 queue time、TTFT、TPOT、ITL、E2E latency；
- prompt/computed/cached/local cached/external cached token 数；
- 聚合 output throughput 和 token 级 KVCache hit rate；
- 每个 scheduler step 的 batch、prefill/decode token、逻辑耗时、KV 占用和命中统计；
- vLLM 版本与完整 latency profile，避免混淆未校准和已校准实验。

## 运行

在仓库根目录执行：

```bash
PYTHONPATH=hisim/src conda run -n hisim-vllm023 \
  env PYTHONHASHSEED=0 VLLM_LOGGING_LEVEL=ERROR \
  python -m hisim.simulation.vllm \
  --model /absolute/path/to/local/model \
  --workload /absolute/path/to/workload.jsonl \
  --output /absolute/path/to/simulation.json \
  --instance-id replica-0
```

### 完整 OpenAI server

```bash
PYTHONPATH=hisim/src conda run -n hisim-vllm023 env \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m hisim.simulation.vllm.launch_server \
  /path/to/local/Qwen3-0.6B \
  --hisim-instance-id=replica-0 \
  --hisim-latency-profile=/path/to/rtx4090_worker_profile.json \
  --hisim-execution-mode=wall_clock \
  --hisim-trace-path=/tmp/hisim-vllm-worker.jsonl \
  --host=127.0.0.1 --port=18001 \
  --served-model-name=qwen3-hisim \
  --max-model-len=512 --max-num-batched-tokens=512 \
  --block-size=16
```

该入口支持标准 `/v1/completions`、`/v1/chat/completions`、SSE streaming、`/metrics` 和
原生 vLLM prefix-cache counters。`wall_clock` 按预测时延 sleep，适合透明服务测试；
`fast_forward` 不等待，用于测量框架空载开销和快速功能回归。

### 原生 offline API

```python
from hisim.simulation.vllm import create_hisim_llm

llm = create_hisim_llm(
    model="/path/to/local/Qwen3-0.6B",
    instance_id="replica-0",
    latency_profile_path="/path/to/profile.json",
    execution_mode="fast_forward",
)
outputs = llm.generate(...)
```

返回对象就是 `vllm.LLM`，可以使用其标准 `generate()`、`chat()` 和
`reset_prefix_cache()` 接口。

`PYTHONHASHSEED=0` 固定 vLLM prefix block hash 的初始化，模型路径必须指向本地缓存，
不需要也不应触发下载。运行单测：

```bash
PYTHONPATH=hisim/src conda run -n hisim-vllm023 \
  env PYTHONHASHSEED=0 VLLM_LOGGING_LEVEL=ERROR \
  python -m pytest -q hisim/test/test_simulation_vllm_scheduler.py
```

## HiSim BenchmarkRunner 接口

`VllmBenchmarkRunner` 与现有 `SGLangBenchmarkRunner` 使用同一调用形态：

```python
from hisim.dataset import SimpleDataset
from hisim.simulation.types import BenchmarkConfig
from hisim.simulation.vllm import VllmBenchmarkRunner

runner = VllmBenchmarkRunner(
    model="/path/to/local/Qwen3-0.6B",
    instance_id="replica-0",
    latency_profile_path="/path/to/rtx4090_profile.json",
)
metrics = runner.benchmark(BenchmarkConfig(), dataset=SimpleDataset(reqs=[...]))
runner.flush_cache()
runner.shutdown()
```

它接受 `BaseDataset` 或 `DatasetArgs`，并返回与 SGLang 路径同名的 `completed`、
`mean_ttft_ms`、`mean_tpot_ms`、`mean_e2e_latency_ms`、`output_throughput` 和
`prefix_cache_reused_ratio`，同时增加 vLLM 原生 queries/hits。一个 runner 固定对应一个
`instance_id`；cache 跨 `benchmark()` 调用保留，但绝不跨 runner/Instance 共享。

两条路径的实现边界：

| 能力 | SGLang backend | vLLM 0.23 backend |
|---|---|---|
| 调度与 KV block 行为 | 真实 SGLang scheduler + hook | 真实 vLLM Scheduler/KVCacheManager |
| 模型 kernel | HiSim mock/hook | 固定 token runner + calibrated profile |
| HiSim Dataset/BenchmarkConfig | 支持 | 支持 |
| cold/warm/flush cache | 支持 | 支持 |
| 统一 TTFT/TPOT/吞吐/命中指标 | 支持 | 支持 |
| 原生 offline engine | 支持 | 支持，返回 `vllm.LLM` |
| 模拟 HTTP server | 支持 | 支持完整 vLLM OpenAI server |
| 框架原生 metrics | SGLang metrics | vLLM Prometheus metrics + Worker trace |
| 无 GPU/权重运行 | 支持 | 支持，不分配真实 KV tensor |

## 校准闭环

1. 在 RTX 4090 上以固定 Qwen 本地模型、input/output length、并发度和 prefix overlap
   运行真实 vLLM 服务。
2. 保存客户端逐请求 TTFT/TPOT/ITL/E2E 与 `/metrics` 中 prefix cache queries/hits；同时
   记录 GPU、驱动、CUDA、Python、vLLM、Torch、命令行和 workload hash。
3. 从 scheduler step trace 与真实批次测量拟合 prefill/decode 延迟 profile，训练集和验证集
   分离。
4. 用同一 token workload 运行真实服务和本适配器，对齐 TTFT、TPOT、吞吐、命中率，报告
   MAPE、P50/P90/P99 误差和不能匹配的原因。
5. 用 `fast_forward` 完整 server 测量 EngineCore/frontend 空载开销，从真实服务时延中扣除后
   拟合 Worker 执行 profile，避免重复计算框架开销。

当前完整 hook 主要验证 Qwen3 dense full-attention generation、TP=PP=1。多模态、LoRA、
speculative decoding、混合 attention/Mamba 和多 rank worker 尚未声明支持；遇到这些配置应扩展
KV spec 和模拟输出，而不能默认外推。
