import importlib.metadata
import gc
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


MODEL_PATH = os.environ.get("HISIM_TEST_VLLM_MODEL_PATH")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str) -> str:
    with urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _completion(base_url: str, request_id: str) -> dict:
    payload = json.dumps(
        {
            "model": "qwen3-hisim-test",
            "request_id": request_id,
            "prompt": [9707] * 48,
            "add_special_tokens": False,
            "max_tokens": 4,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _chat_completion(base_url: str) -> dict:
    payload = json.dumps(
        {
            "model": "qwen3-hisim-test",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 2,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.skipif(MODEL_PATH is None, reason="set HISIM_TEST_VLLM_MODEL_PATH")
@pytest.mark.skipif(
    importlib.metadata.version("vllm") != "0.23.0", reason="requires vLLM 0.23.0"
)
def test_full_vllm_server_uses_hisim_worker_and_native_prefix_cache(tmp_path):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    trace_path = tmp_path / "worker-trace.jsonl"
    log_path = tmp_path / "server.log"
    profile_path = tmp_path / "latency-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "latency_profile": {
                    "name": "integration-test",
                    "scheduler_overhead_ms": 0,
                    "prefill_base_ms": 20,
                    "prefill_token_ms": 0,
                    "decode_base_ms": 5,
                    "decode_token_ms": 0,
                    "decode_context_token_ms": 0,
                    "calibrated": False,
                }
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "hisim.simulation.vllm.launch_server",
        MODEL_PATH,
        "--hisim-instance-id=integration-test",
        "--hisim-execution-mode=wall_clock",
        f"--hisim-latency-profile={profile_path}",
        f"--hisim-trace-path={trace_path}",
        "--hisim-num-gpu-blocks=128",
        "--host=127.0.0.1",
        f"--port={port}",
        "--served-model-name=qwen3-hisim-test",
        "--max-model-len=256",
        "--max-num-batched-tokens=256",
        "--max-num-seqs=8",
        "--block-size=16",
        "--enable-prompt-tokens-details",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "VLLM_LOGGING_LEVEL": "WARNING",
        }
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text(encoding="utf-8"))
                try:
                    _get(f"{base_url}/health")
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError("vLLM HiSim server did not become ready")

            cold = _completion(base_url, "cold")
            warm = _completion(base_url, "warm")
            chat = _chat_completion(base_url)
            with ThreadPoolExecutor(max_workers=4) as executor:
                concurrent_outputs = list(
                    executor.map(
                        lambda index: _completion(base_url, f"concurrent-{index}"),
                        range(4),
                    )
                )
            metrics = _get(f"{base_url}/metrics")

            assert cold["choices"][0]["text"] == "Hello" * 4
            assert warm["choices"][0]["text"] == "Hello" * 4
            assert chat["choices"][0]["message"]["content"] == "Hello" * 2
            assert all(
                output["choices"][0]["text"] == "Hello" * 4
                for output in concurrent_outputs
            )
            assert cold["usage"].get("prompt_tokens_details") is None
            assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] >= 32
            assert 'vllm:prefix_cache_hits_total{engine="0"' in metrics
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            assert trace[0]["instance_id"] == "integration-test"
            assert any(record["event"] == "execute_model" for record in trace)
            assert any(
                len(record.get("request_ids", [])) > 1 for record in trace
            ), "concurrent requests were not batched by the native vLLM scheduler"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


@pytest.mark.skipif(MODEL_PATH is None, reason="set HISIM_TEST_VLLM_MODEL_PATH")
@pytest.mark.skipif(
    importlib.metadata.version("vllm") != "0.23.0", reason="requires vLLM 0.23.0"
)
def test_native_vllm_offline_api_uses_hisim_worker(tmp_path):
    from hisim.simulation.vllm import create_hisim_llm

    trace_path = tmp_path / "offline-trace.jsonl"
    llm = create_hisim_llm(
        model=MODEL_PATH,
        instance_id="offline-integration-test",
        execution_mode="fast_forward",
        trace_path=trace_path,
        num_gpu_blocks=128,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=8,
    )
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
    prompt = "KV cache simulation " * 20
    cold = llm.generate([prompt], params, use_tqdm=False)[0]
    warm = llm.generate([prompt], params, use_tqdm=False)[0]

    assert cold.outputs[0].text == "Hello" * 4
    assert warm.outputs[0].text == "Hello" * 4
    assert cold.num_cached_tokens == 0
    assert warm.num_cached_tokens >= 16
    assert llm.reset_prefix_cache()
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert trace[0]["instance_id"] == "offline-integration-test"
    del llm
    gc.collect()
