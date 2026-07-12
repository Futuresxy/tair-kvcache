# HiSim / vLLM 实验环境

本目录记录 HiSim vLLM 接入实验的可复现环境。环境分离是有意设计：

- `hisim`：保留 SGLang 0.5.6.post2、AIConfigurator 和现有 HiSim 回归测试。
- `hisim-vllm023`：固定 vLLM 0.23.0，用于 RTX 4090 真实推理和 vLLM adapter 测试。

两个框架对 `torch`、`flashinfer`、`llguidance`、`outlines-core` 和 `numpy`
的版本要求互相冲突，因此不要把 vLLM 0.23.0 直接升级进 `hisim`。

## 创建 vLLM 环境

推荐从最小 Python 环境按锁定文件安装：

```bash
conda create -n hisim-vllm023 python=3.10 -y
conda run -n hisim-vllm023 python -m pip install -r \
  hisim/envs/hisim-vllm023.requirements.txt
```

当前机器最初通过克隆 `hisim` 后升级 vLLM 创建；随后从 vLLM 环境移除了
`sglang`、`aiconfigurator`、`outlines` 和克隆来的 `hisim` wheel，以保持
`pip check` 干净。仓库源码通过工作区运行，不把本地绝对路径写入锁文件。

## GPU 可见性

原 `hisim` 环境保存了空的 `CUDA_VISIBLE_DEVICES`，`conda run` 会覆盖外部同名
变量。真实 GPU 命令必须在 conda 环境内部覆盖：

```bash
conda run -n hisim-vllm023 env CUDA_VISIBLE_DEVICES=0 \
  python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

在受限执行环境中，新环境访问 GPU 可能需要沙箱外执行。RTX 4090 校验成功时：

```text
torch 2.11.0+cu130
CUDA 13.0
NVIDIA GeForce RTX 4090
```

## 基线环境

`hisim` 的关键版本记录在 `hisim-sglang-baseline.requirements.txt`。2026-07-12
已验证以下测试通过：

```bash
conda run -n hisim python -m pytest -q \
  hisim/test/test_aic_xgb_predictor.py \
  hisim/test/test_simulation_time_predictor.py \
  hisim/test/test_spec_registry.py
```

结果：`4 passed`。

## 环境检查

```bash
conda run -n hisim-vllm023 python -m pip check
conda run -n hisim-vllm023 python -c \
  'import vllm, torch; print(vllm.__version__, torch.__version__)'
```

每次升级依赖或驱动后都应更新 `manifest.json`，并在 commit 中单独说明环境变化。

