import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

tools = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(tools))
spec = importlib.util.spec_from_file_location("serving_audit", tools / "audit_serving.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    prompt = [1, 2]
    identity = {"runtime_profile_sha256": "profile"}
    row = {"condition": "marked", "max_tokens": 16, "completion_tokens": 2,
           "completion": "eos", "text": "Hello", "text_sha256": hashlib.sha256(b"Hello").hexdigest()}
    report = {"target_identity": identity, "payload": {"assigned_condition": "marked",
              "committed_token_ids": [3, 4], "sampling_records": [{"token_id": 3}, {"token_id": 4}],
              "completion": "eos"}, "usage": {"completion_tokens": 2},
              "rendered_carriers": {"visible_text": "Hello"}}
    start = {"kind": "response_started", "prompt_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
             "prompt_length": 2, "condition": "marked", "max_tokens": 16, **identity}
    events = [start, {"kind": "committed_step", "index": 0}, {"kind": "committed_step", "index": 1},
              {"kind": "response_terminal", "sampled_tokens": 2, "journal_broken": False, "error_type": None}]
    return row, report, events, prompt, identity


def test_matching_retained_output_verifies_committed_count():
    assert audit.verify_output(*fixture()) == 2


@pytest.mark.parametrize("change", ["condition", "prompt", "token", "count", "commit_order",
                                  "broken_journal", "missing_terminal", "text"])
def test_output_reconciliation_rejects_mismatched_receipts(change):
    row, report, events, prompt, identity = copy.deepcopy(fixture())
    if change == "condition": row["condition"] = "ordinary"
    if change == "prompt": prompt[0] = 9
    if change == "token": report["payload"]["sampling_records"][0]["token_id"] = 9
    if change == "count": row["completion_tokens"] = 3
    if change == "commit_order": events[1]["index"] = 1
    if change == "broken_journal": events[-1]["journal_broken"] = True
    if change == "missing_terminal": events.pop()
    if change == "text": row["text"] = "Substituted"
    with pytest.raises(ValueError):
        audit.verify_output(row, report, events, prompt, identity)
