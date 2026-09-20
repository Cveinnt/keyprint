import importlib.util
from pathlib import Path
import sys

import pytest

tools = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(tools))
spec = importlib.util.spec_from_file_location("partial_null_audit", tools / "audit_partial_predictability_null.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    rows = [{"source_index": i, "unavailable": False, "family_reference_tail": p,
             "flagged": p <= .01, "baseline_flagged": baseline}
            for i, (p, baseline) in enumerate(((.009, True), (.02, True), (.003, False), (.4, False)))]
    rows.append({"source_index": 4, "error": {"type": "OSError"}})
    summary = {"status": "incomplete", "attempts": 5, "errors": 1, "unavailable": 0,
               "fatal": None, "hits": 2, "baseline_hits": 2, "iid_only_upper_97_5": None,
               "deployment_calibrated": False}
    return rows, summary


def test_partial_replay_preserves_failure_and_same_control_comparison():
    rows, summary = fixture()
    usable, counts = audit.reconcile(rows, summary, set(range(5)))
    assert len(usable) == 4
    assert counts["failed_indices"] == [4]
    assert counts["transitions"] == dict(both_flagged=1, candidate_only=1, baseline_only=1, neither=1)
    assert counts["iid_only_upper_97_5"] is None
    assert counts["deployment_calibrated"] is False


@pytest.mark.parametrize("change", ["drop_failure", "duplicate", "replacement", "false_negative",
                                  "bound", "completed", "bad_count", "unavailable", "nan", "flag"])
def test_partial_evidence_cannot_be_laundered_into_complete_acceptance(change):
    rows, summary = fixture()
    if change == "drop_failure": rows.pop()
    if change == "duplicate": rows[4] = dict(rows[0])
    if change == "replacement": rows[4]["source_index"] = 5
    if change == "false_negative": rows[4]["flagged"] = False
    if change == "bound": summary["iid_only_upper_97_5"] = .03
    if change == "completed": summary["status"] = "completed"
    if change == "bad_count": summary["hits"] = 1
    if change == "unavailable": rows[0]["unavailable"] = True
    if change == "nan": rows[0]["family_reference_tail"] = float("nan")
    if change == "flag": rows[0]["flagged"] = False
    with pytest.raises(ValueError):
        audit.reconcile(rows, summary, set(range(5)))
