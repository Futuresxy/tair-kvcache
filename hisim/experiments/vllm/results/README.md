# 结果文件索引

最终可复现的单并发校准闭环：

- `train_grid_*_server_real.json`：64/128/192-token 训练网格的 RTX 4090 原始数据；
- `rtx4090_qwen3_0.6b_bs1_server_profile_v3.json`：从训练网格拟合的固定 profile；
- `heldout_grid_*_server_real.json`：96/160-token held-out RTX 4090 原始数据；
- `heldout_grid_*_server_simulation.json`：同 token workload 的 vLLM scheduler 仿真；
- `heldout_grid_*_server_comparison.json`：逐请求与聚合误差。

文件名中没有 `server` 的 JSON 是开发过程中保留的诊断证据：它们使用客户端 SSE 时间，包含
HTTP/连接抖动。`bs1_profile`、`grid_profile_v2` 及相应 comparison 是被 held-out 实验否定的
早期 profile。保留这些文件是为了让校准决策可审计，不应作为最终精度结果引用。

所有 real JSON 都包含精确 prompt token IDs、GPU/驱动、Python、vLLM、Torch、Transformers
版本和原始 Prometheus counter delta；profile 记录输入文件 SHA-256。

完整 Engine/Worker hook 的最终工件：

- `full_hook_overhead_*`：`fast_forward` 完整 server 的框架空载开销；
- `rtx4090_qwen3_0.6b_full_engine_worker_profile_v4.json`：扣除空载开销后的 Worker profile；
- `full_hook_*_server_v4.json`：完整 HiSim OpenAI server 的 held-out 结果；
- `full_hook_*_comparison_v4.json`：与真实 RTX 4090 vLLM server 的逐请求对比。

不带 `_v4` 的 `full_hook` 结果使用了包含框架开销的早期 profile，保留用于审计双重计算时延问题。

`tiered_kv_long_context_validation.json` 记录 4096-token 长上下文的多级缓存功能验证：在原始
prefix 已从 HBM 淘汰后，vLLM 原生 Scheduler 从 SSD 接受 4080 个 external cached tokens，
只保留 16 token 重计算。该文件中的 cost 参数是策略演示值，不是 RTX 4090 精度校准结论。
