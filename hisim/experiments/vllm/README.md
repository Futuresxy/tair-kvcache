# RTX 4090 / vLLM 0.23 校准实验

实验固定使用隔离环境 `hisim-vllm023`、vLLM 0.23.0 和本地模型目录。所有命令都设置
`HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，禁止测试过程中下载模型。

启动服务：

```bash
conda run -n hisim-vllm023 env \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONHASHSEED=0 \
  vllm serve /path/to/local/Qwen3-0.6B \
  --host 127.0.0.1 --port 18000 \
  --served-model-name qwen3-0.6b \
  --max-model-len 512 --gpu-memory-utilization 0.30 \
  --enforce-eager --enable-prefix-caching
```

采集真实服务指标：

```bash
conda run -n hisim-vllm023 env \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python hisim/experiments/vllm/benchmark_openai.py \
  --tokenizer /path/to/local/Qwen3-0.6B \
  --output hisim/experiments/vllm/results/rtx4090_real.json
```

脚本发送一个 cold 请求和若干完全相同的 warm 请求，保留精确 prompt token IDs，采集客户端
streaming TTFT/TPOT/E2E，以及服务 `/metrics` 的 `vllm:prefix_cache_queries_total` 和
`vllm:prefix_cache_hits_total` 差值。结果中的 prompt token IDs 将直接转换成 HiSim workload，
确保真实服务与仿真使用相同输入。

校准主口径使用服务端 Prometheus histogram 的逐请求 delta：

- `vllm:time_to_first_token_seconds`；
- `vllm:request_time_per_output_token_seconds`；
- `vllm:inter_token_latency_seconds`；
- `vllm:e2e_request_latency_seconds`。

客户端 SSE 时间仍保存在 `client_*` 字段，但不用于 scheduler latency profile 拟合。这样可以把
HTTP 连接与前端抖动作为独立层分析。

首版线性 profile 与同 workload 回放：

```bash
PYTHONPATH=hisim/src conda run -n hisim-vllm023 \
  env PYTHONHASHSEED=0 VLLM_LOGGING_LEVEL=ERROR \
  python hisim/experiments/vllm/calibrate_scheduler.py \
  --real hisim/experiments/vllm/results/rtx4090_real.json \
  --model-config /path/to/local/Qwen3-0.6B \
  --profile-out hisim/experiments/vllm/results/rtx4090_profile.json \
  --workload-out hisim/experiments/vllm/results/workload.jsonl \
  --simulation-out hisim/experiments/vllm/results/simulation.json \
  --comparison-out hisim/experiments/vllm/results/comparison.json
```

该步骤是 `in_sample_fit`，只验证真实 vLLM 调度语义、指标口径和校准管线闭环；它不能作为
仿真精度结论。精度结论必须来自不同 input/output length、并发度和命中率的 held-out workload。

## 当前 RTX 4090 单并发结果

固定环境为 vLLM 0.23.0、Torch 2.11.0+cu130、本地 Qwen3-0.6B、block size 16。
`rtx4090_qwen3_0.6b_bs1_server_profile_v3.json` 使用 64/128/192-token 三种训练长度，
每种长度包含 3 个互不共享 prefix 的 cold/warm 对。

| held-out workload | TTFT MAPE | TPOT MAPE | E2E MAPE | active throughput error | KV hit-rate abs error |
|---|---:|---:|---:|---:|---:|
| prefix 96, output 16 | 14.95% | 6.29% | 6.87% | 1.36% | 0.00 |
| prefix 160, output 24 | 7.87% | 3.50% | 2.70% | 2.80% | 0.00 |

这些是未参与拟合的服务端计时结果，只覆盖 batch size 1。早期客户端计时与单点拟合结果也保留
在 `results/` 中，证明为何需要分离 HTTP 抖动和 engine timing；不能用它们替代最终 v3 结果。

## 完整 Engine/Worker hook 结果

完整 server 自身会引入 EngineCore、IPC、output processing 和 frontend 开销。使用
`fast_forward` server 在相同训练 workload 上测量这部分空载开销，并通过
`fit_full_hook_profile.py` 从真实 RTX 4090 指标中扣除，得到
`rtx4090_qwen3_0.6b_full_engine_worker_profile_v4.json`。

| held-out workload | TTFT MAPE | TPOT MAPE | E2E MAPE | active throughput error | KV hit-rate abs error |
|---|---:|---:|---:|---:|---:|
| prefix 96, output 16 | 9.01% | 6.98% | 7.13% | 0.62% | 0.00 |
| prefix 160, output 24 | 5.21% | 2.45% | 2.59% | 2.69% | 0.00 |

这些结果通过完整 OpenAI server 获得，不是直接调用 standalone scheduler。真实服务和 HiSim
服务使用完全相同的 prompt token IDs、vLLM Scheduler 配置和 prefix-cache 指标口径。
