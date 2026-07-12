from pathlib import Path

import pytest

from hisim.simulation.vllm import VllmRequestSpec, VllmSchedulerSimulator


MODEL_CONFIG = Path(__file__).parent / "assets" / "vllm_tiny_qwen"


def make_simulator(instance_id: str = "instance-a") -> VllmSchedulerSimulator:
    return VllmSchedulerSimulator(
        model=str(MODEL_CONFIG),
        instance_id=instance_id,
        block_size=16,
        num_gpu_blocks=128,
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=8,
    )


def test_vllm_scheduler_completes_and_reports_latency_metrics():
    simulator = make_simulator()
    result = simulator.run(
        [
            VllmRequestSpec(
                request_id="request-0",
                prompt_token_ids=list(range(33)),
                max_tokens=4,
                instance_id="instance-a",
            )
        ]
    )

    request = result.requests[0]
    assert request.output_tokens == 4
    assert request.ttft_ms is not None and request.ttft_ms > 0
    assert request.tpot_ms is not None and request.tpot_ms > 0
    assert request.e2e_latency_ms is not None
    assert result.output_throughput_tokens_per_s > 0
    assert result.to_dict()["vllm_version"] == "0.23.0"


def test_vllm_prefix_cache_reuses_full_blocks_for_later_request():
    simulator = make_simulator()
    shared_prompt = list(range(48))
    first = simulator.run(
        [
            VllmRequestSpec(
                request_id="cold",
                prompt_token_ids=shared_prompt,
                max_tokens=2,
                instance_id="instance-a",
            )
        ]
    )
    second = simulator.run(
        [
            VllmRequestSpec(
                request_id="warm",
                prompt_token_ids=shared_prompt,
                max_tokens=2,
                instance_id="instance-a",
            )
        ]
    )

    assert first.requests[0].cached_tokens == 0
    assert second.requests[0].local_cached_tokens >= 16
    assert second.requests[0].cached_tokens == second.requests[0].local_cached_tokens
    assert second.kv_cache_hit_rate > 0


def test_vllm_simulator_rejects_cross_instance_request():
    simulator = make_simulator(instance_id="instance-a")

    with pytest.raises(ValueError, match="instance isolation"):
        simulator.run(
            [
                VllmRequestSpec(
                    request_id="wrong-instance",
                    prompt_token_ids=list(range(16)),
                    max_tokens=1,
                    instance_id="instance-b",
                )
            ]
        )
