import importlib.util
import json
import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
CONVERTER_PATH = TEST_DIR / "scripts/llmservingsim_to_hisim.py"
SAMPLE_PATH = TEST_DIR / "assets/llmservingsim/llmservingsim_mixed_sample.jsonl"


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "llmservingsim_to_hisim", CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_llmservingsim_converter_cli(tmp_path):
    output_path = tmp_path / "converted.hisim.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_PATH),
            "-i",
            str(SAMPLE_PATH),
            "-o",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Converted 3 requests" in result.stdout
    assert "Skipped 1 requests without input_tok_ids" in result.stdout

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["rid"] for record in records] == [
        "flat-0",
        "task-0-run0-sub0",
        "task-0-run0-sub1",
    ]
    assert [record["timestamp"] for record in records] == [0.0, 1.0, 2.5]
    assert [record["input_length"] for record in records] == [5, 3, 5]
    assert [record["output_length"] for record in records] == [2, 1, 2]
    assert records[1]["input_ids"] == [10, 11, 12]
    assert records[2]["input_ids"] == [10, 11, 12, 13, 14]

    converter = load_converter()
    assert converter.best_prior_prefix_lens(records) == [0, 0, 3]


def test_llmservingsim_converter_synthetic_missing_ids(tmp_path):
    output_path = tmp_path / "converted.synthetic.hisim.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(CONVERTER_PATH),
            "-i",
            str(SAMPLE_PATH),
            "-o",
            str(output_path),
            "--synthetic-missing-token-ids",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 4
    assert records[-1]["rid"] == "flat-2"
    assert records[-1]["input_length"] == 4
    assert len(records[-1]["input_ids"]) == 4
