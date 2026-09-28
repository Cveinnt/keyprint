import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from audit_wide_mlx import audit_draw, audit_steps


def events():
    return [{"phase": "start"}, {"phase": "model_forward", "index": 0},
            {"phase": "prepared", "raw_logits_sha256": "a" * 64, "weights_sha256": "b" * 64},
            {"phase": "random_requested", "bits": 2}, {"phase": "random_returned", "bits": 2, "value": 3},
            {"phase": "random_requested", "bits": 2}, {"phase": "random_returned", "bits": 2, "value": 1},
            {"phase": "commit_requested", "token_id": 248319, "draw": {"token_index": 248319,
             "integer_point": 1, "total_weight": 3, "transcript": [
                 {"bit_count": 2, "value": 3, "accepted": False}, {"bit_count": 2, "value": 1, "accepted": True}]}},
            {"phase": "committed", "token_id": 248319}, {"phase": "complete"}]


def test_high_id_and_rejected_draw_are_reconciled():
    assert len(audit_steps(events(), [248319])) == 64
    audit_draw({"total_weight": 1, "integer_point": 0, "transcript": []}, [])


@pytest.mark.parametrize("mutation", ["token", "point", "accepted", "returned", "bits", "forward", "extra", "terminal"])
def test_tampered_receipts_are_rejected(mutation):
    rows = copy.deepcopy(events())
    if mutation == "token": rows[8]["token_id"] = 0
    elif mutation == "point": rows[7]["draw"]["integer_point"] = 2
    elif mutation == "accepted": rows[7]["draw"]["transcript"][0]["accepted"] = True
    elif mutation == "returned": rows[6]["value"] = 2
    elif mutation == "bits": rows[3]["bits"] = 3
    elif mutation == "forward": rows[1]["index"] = 1
    elif mutation == "extra": rows.insert(-1, {"phase": "model_forward", "index": 1})
    elif mutation == "terminal": rows.pop()
    with pytest.raises(ValueError):
        audit_steps(rows, [248319])
