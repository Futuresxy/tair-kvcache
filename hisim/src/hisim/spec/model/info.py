from hisim.spec.model.base import ModelInfo


def _register(config: dict, *aliases: str) -> ModelInfo:
    model = ModelInfo.from_dict(config)
    if model is None:
        raise ValueError(f"Invalid built-in model config: {config.get('name')}")
    ModelInfo.register_model(model, *aliases)
    return model


class Qwen:
    QWEN2_5_7B_INSTRUCT = _register(
        {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": "qwen2",
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen2ForCausalLM",
        },
        "Qwen2.5-7B",
        "Qwen2.5-7B-Instruct",
    )

    QWEN2_5_32B_INSTRUCT = _register(
        {
            "name": "Qwen/Qwen2.5-32B-Instruct",
            "model_type": "qwen2",
            "hidden_size": 5120,
            "intermediate_size": 27648,
            "num_hidden_layers": 64,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen2ForCausalLM",
        },
        "Qwen2.5-32B",
        "Qwen2.5-32B-Instruct",
    )

    QWEN3_0_6B = _register(
        {
            "name": "Qwen/Qwen3-0.6B",
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3ForCausalLM",
        },
        "Qwen3-0.6B",
    )

    QWEN3_1_7B = _register(
        {
            "name": "Qwen/Qwen3-1.7B",
            "model_type": "qwen3",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3ForCausalLM",
        },
        "Qwen3-1.7B",
    )

    QWEN3_4B = _register(
        {
            "name": "Qwen/Qwen3-4B",
            "model_type": "qwen3",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3ForCausalLM",
        },
        "Qwen3-4B",
    )

    QWEN3_8B = _register(
        {
            "name": "Qwen/Qwen3-8B",
            "model_type": "qwen3",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3ForCausalLM",
        },
        "Qwen3-8B",
    )

    QWEN3_14B = _register(
        {
            "name": "Qwen/Qwen3-14B",
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3ForCausalLM",
        },
        "Qwen3-14B",
    )

    QWEN3_30B_A3B = _register(
        {
            "name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3MoeForCausalLM",
        },
        "Qwen3-30B-A3B",
    )

    QWEN3_32B = _register(
        {
            "name": "Qwen/Qwen3-32B",
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3ForCausalLM",
        },
        "Qwen3-32B",
        "Qwen/Qwen3-32B-FP8",
        "Qwen3-32B-FP8",
    )

    QWEN3_235B_A22B = _register(
        {
            "name": "Qwen/Qwen3-235B-A22B",
            "model_type": "qwen3_moe",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1536,
            "num_hidden_layers": 94,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3MoeForCausalLM",
        },
        "Qwen3-235B-A22B",
        "Qwen/Qwen3-235B-A22B-FP8",
        "Qwen3-235B-A22B-FP8",
    )

    QWEN3_CODER_480B_A35B = _register(
        {
            "name": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "model_type": "qwen3_moe",
            "hidden_size": 6144,
            "intermediate_size": 8192,
            "moe_intermediate_size": 2560,
            "num_hidden_layers": 62,
            "num_attention_heads": 96,
            "num_key_value_heads": 8,
            "num_experts": 160,
            "num_experts_per_tok": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "architecture": "Qwen3MoeForCausalLM",
        },
        "Qwen3-Coder-480B-A35B",
        "Qwen3-Coder-480B-A35B-Instruct",
    )


class DeepSeek:
    DEEPSEEK_V3 = _register(
        {
            "name": "deepseek-ai/DeepSeek-V3",
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 3,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "torch_dtype": "bfloat16",
            "architecture": "DeepseekV3ForCausalLM",
            "quantization_config": {"quant_method": "fp8"},
        },
        "DeepSeek-V3",
        "DeepSeekV3",
    )

    DEEPSEEK_R1 = _register(
        {
            "name": "deepseek-ai/DeepSeek-R1",
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 3,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "torch_dtype": "bfloat16",
            "architecture": "DeepseekV3ForCausalLM",
            "quantization_config": {"quant_method": "fp8"},
        },
        "DeepSeek-R1",
        "DeepSeekR1",
    )
