from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo


def test_accelerator_registry_contains_common_hardware():
    cases = {
        "h20": ("NVIDIA", 96),
        "h100_sxm": ("NVIDIA", 80),
        "H200_SXM": ("NVIDIA", 141),
        "RTX4090": ("NVIDIA", 24),
        "ascend910b": ("Huawei", 64),
        "Ascend950": ("Huawei", 144),
    }

    for alias, (vendor, memory_gb) in cases.items():
        hw = AcceleratorInfo.find_by_hw_name(alias)
        assert hw is not None, alias
        assert hw.vendor == vendor
        assert hw.hbm_capacity_gb == memory_gb


def test_model_registry_contains_common_llms():
    qwen = ModelInfo.find_by_model_name("Qwen/Qwen3-32B")
    assert qwen is not None
    assert qwen.model_type == "qwen3"
    assert qwen.num_hidden_layers == 64
    assert qwen.num_key_value_heads == 8

    qwen_fp8_alias = ModelInfo.find_by_model_name("Qwen3-32B-FP8")
    assert qwen_fp8_alias is not None
    assert qwen_fp8_alias.hidden_size == qwen.hidden_size

    deepseek = ModelInfo.find_by_model_name("deepseek-ai/DeepSeek-R1")
    assert deepseek is not None
    assert deepseek.model_type == "deepseek_v3"
    assert deepseek.kv_lora_rank == 512
    assert deepseek.n_routed_experts == 256
