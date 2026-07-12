# vLLM 0.23 接入设计与复现实验入口

## 当前边界

第一阶段适配器直接实例化 vLLM 0.23.0 的 `Scheduler`、`KVCacheManager`、
block allocator 和 prefix block hasher。HiSim 只替换模型执行器：每个可解码请求产生一个
固定 token，并使用可替换的延迟模型推进逻辑时钟。因此调度、分块、前缀复用、抢占和结束判定
来自真实 vLLM 代码，TTFT、TPOT、吞吐和 KVCache 命中指标由同一次事件循环计算。

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

`PYTHONHASHSEED=0` 固定 vLLM prefix block hash 的初始化，模型路径必须指向本地缓存，
不需要也不应触发下载。运行单测：

```bash
PYTHONPATH=hisim/src conda run -n hisim-vllm023 \
  env PYTHONHASHSEED=0 VLLM_LOGGING_LEVEL=ERROR \
  python -m pytest -q hisim/test/test_simulation_vllm_scheduler.py
```

## 后续校准闭环

1. 在 RTX 4090 上以固定 Qwen 本地模型、input/output length、并发度和 prefix overlap
   运行真实 vLLM 服务。
2. 保存客户端逐请求 TTFT/TPOT/ITL/E2E 与 `/metrics` 中 prefix cache queries/hits；同时
   记录 GPU、驱动、CUDA、Python、vLLM、Torch、命令行和 workload hash。
3. 从 scheduler step trace 与真实批次测量拟合 prefill/decode 延迟 profile，训练集和验证集
   分离。
4. 用同一 token workload 运行真实服务和本适配器，对齐 TTFT、TPOT、吞吐、命中率，报告
   MAPE、P50/P90/P99 误差和不能匹配的原因。
5. 校准通过后再将该 backend 接到 HiSim 的统一 benchmark/manager 入口，并与现有 SGLang
   接入使用同一结果 schema 做回归。
