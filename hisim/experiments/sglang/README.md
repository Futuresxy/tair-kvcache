# SGLang 多级 KVCache 实验

该目录验证固定 SGLang 0.5.6.post2 经真实 Engine/Scheduler/HiCache 控制流接入 HiSim 共享
HBM/DRAM/SSD 策略。详细语义见 `hisim/docs/sglang_tiered_cache.md`。

`rtx4090_qwen3_0.6b_tiered_kv_example.json` 是 `cost_aware` 示例；
`tiered_kv_always_validation.json` 用于强制读取、验证部分预取和运行时命中统计。

```bash
PYTHONPATH=hisim/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n hisim python \
  hisim/experiments/sglang/validate_tiered_engine.py \
  --model /absolute/path/to/local/Qwen3-0.6B \
  --simulation-config hisim/experiments/sglang/rtx4090_qwen3_0.6b_tiered_kv_example.json \
  --tiered-config hisim/experiments/sglang/tiered_kv_always_validation.json \
  --output /tmp/hisim/sglang-tiered-validation/always.json
```

使用 `cost_aware` 时把两个 config 参数都指向
`rtx4090_qwen3_0.6b_tiered_kv_example.json`。脚本设置严格 Instance namespace，运行冷/暖两轮，
且不访问 Hugging Face 网络。
