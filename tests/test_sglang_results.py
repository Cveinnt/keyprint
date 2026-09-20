import copy
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_native_receipts import check as check_tokens
from check_sglang_results import check
from keyprint.experimental.sglang_runtime import REVISION, SOURCE_FILES, VERSIONS
from keyprint.sampling import identity as sampling_identity


@pytest.fixture
def receipts(tmp_path):
    (tmp_path / "public").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "run-id.txt").write_text("test-run\n")
    for name in ("exit-code.txt", "contract-exit-code.txt"):
        (tmp_path / name).write_text("0\n")
    cases = json.loads((Path(__file__).resolve().parents[1] / "tools/inference_cases.json").read_text())
    rows, outputs = [], []
    for case in cases:
        for condition in ("ordinary", "marked"):
            token = len(rows) + 1
            output = {"text": "Synthetic contract fixture", "token_ids": [token], "completion": "eos"}
            outputs.append(output)
            rows.append({"case": case["id"], "condition": condition, **output,
                         "usage": {"completion_tokens": 1}})
            events = [{"phase": "start", "condition": condition,
                       "adapter_sha256": hashlib.sha256(files("keyprint").joinpath("experimental/native.py").read_bytes()).hexdigest(),
                       "sampling_execution": sampling_identity()},
                      {"phase": "selected_tentative", "token_id": token}]
            previous, lines = "0" * 64, []
            for sequence, event in enumerate(events):
                line = (json.dumps({"sequence": sequence, "previous_sha256": previous, "event": event}) + "\n").encode()
                lines.append(line)
                previous = hashlib.sha256(line).hexdigest()
            (tmp_path / f"traces/{token}.jsonl").write_bytes(b"".join(lines))
    (tmp_path / "outputs.json").write_text(json.dumps(outputs))
    report = {"backend": "sglang CPU pilot", "cases": cases, "runs": rows, "engineering_failures": [],
              "runtime_identity": {"source_revision": REVISION, "build_version": sorted(VERSIONS)[0],
                                   "source_files_sha256": SOURCE_FILES},
              "token_path_verification": check_tokens(tmp_path)}
    (tmp_path / "public/comparison.json").write_text(json.dumps(report))
    return tmp_path


def test_complete_receipts_replayed_without_model_dependencies(receipts):
    assert check(receipts, "test-run")["requests"] == 12


def test_cached_run_rejected(receipts):
    with pytest.raises(ValueError, match="run ID mismatch"):
        check(receipts, "new-run")


@pytest.mark.parametrize("name", ["exit-code.txt", "contract-exit-code.txt"])
def test_failure_or_missing_exit_rejected(receipts, name):
    (receipts / name).write_text("137")
    with pytest.raises(ValueError, match="not zero"):
        check(receipts, "test-run")
    (receipts / name).unlink()
    with pytest.raises(FileNotFoundError):
        check(receipts, "test-run")


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d["runs"].pop(), "complete ordinary"),
    (lambda d: d["runs"].__setitem__(1, copy.deepcopy(d["runs"][0])), "duplicate"),
    (lambda d: d["cases"][0].update(prompt="easier prompt"), "inference cases differ"),
    (lambda d: d["runtime_identity"].update(source_files_sha256={}), "runtime identity differs"),
    (lambda d: d["runs"][0].update(text="changed after generation"), "actual outputs"),
    (lambda d: d["token_path_verification"].update(matching_tokens=9000), "replayed journals"),
])
def test_partial_or_altered_results_rejected(receipts, mutation, message):
    path = receipts / "public/comparison.json"
    report = json.loads(path.read_text())
    mutation(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        check(receipts, "test-run")


def test_changed_journal_rejected_even_when_report_claims_pass(receipts):
    path = receipts / "traces/1.jsonl"
    path.write_text(path.read_text().replace('"token_id": 1', '"token_id": 900'))
    with pytest.raises(ValueError, match="token path"):
        check(receipts, "test-run")


def test_swapping_ordinary_and_marked_labels_rejected(receipts):
    path = receipts / "public/comparison.json"
    report = json.loads(path.read_text())
    report["runs"][0]["condition"], report["runs"][1]["condition"] = "marked", "ordinary"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="displayed condition"):
        check(receipts, "test-run")


@pytest.mark.parametrize("field,value,message", [
    ("adapter_sha256", None, "adapter source"),
    ("adapter_sha256", "0" * 64, "adapter source"),
    ("sampling_execution", None, "sampling execution"),
    ("sampling_execution", {**sampling_identity(), "source_sha256": "0" * 64}, "sampling execution"),
])
def test_old_or_missing_adapter_identity_rejected_even_with_consistent_receipts(receipts, field, value, message):
    path = receipts / "traces/1.jsonl"
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    if value is None:
        events[0].pop(field)
    else:
        events[0][field] = value
    previous, lines = "0" * 64, []
    for sequence, event in enumerate(events):
        line = (json.dumps({"sequence": sequence, "previous_sha256": previous, "event": event}) + "\n").encode()
        lines.append(line)
        previous = hashlib.sha256(line).hexdigest()
    path.write_bytes(b"".join(lines))
    report_path = receipts / "public/comparison.json"
    report = json.loads(report_path.read_text())
    report["token_path_verification"] = check_tokens(receipts)
    report_path.write_text(json.dumps(report))
    # Tokens, conditions, hash chains and the report all agree. Only the source
    # identity is stale/missing; a matching token path cannot qualify that code.
    with pytest.raises(ValueError, match=message):
        check(receipts, "test-run")
