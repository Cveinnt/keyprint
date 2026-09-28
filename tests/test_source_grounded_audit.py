import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from audit_source_grounded import audit_output, reconcile
from validate_source_grounded import schedule


def study():
    cases = [{"id": str(i), "prompt": f"source {i}"} for i in range(16)]
    rubrics = [{"id": str(i), "essential_facts": [str(i)]} for i in range(16)]
    plan = {"schedule": schedule(cases)}
    rows = [dict(r, review_id=f"{i:012x}", text="output", completion="eos")
            for i, r in enumerate(plan["schedule"])]
    review = [{"review_id": r["review_id"], "case": cases[int(r["case"])],
               "rubric": rubrics[int(r["case"])], "text": r["text"],
               "completion": r["completion"], "error_type": None} for r in rows]
    return plan, cases, rubrics, rows, review


def test_all_attempts_and_shuffled_review_match():
    args = study()
    args[-1].reverse()
    reconcile(*args)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reorder", "source", "rubric", "text", "leak", "review_missing", "id"])
def test_selective_or_tampered_review_rejected(change):
    plan, cases, rubrics, rows, review = copy.deepcopy(study())
    if change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = rows[0]
    elif change == "reorder": rows.reverse()
    elif change == "source": review[0]["case"] = {"id": "0", "prompt": "changed source"}
    elif change == "rubric": review[0]["rubric"] = {"id": "0", "essential_facts": []}
    elif change == "text": review[0]["text"] = "repaired output"
    elif change == "leak": review[0]["condition"] = "ordinary"
    elif change == "review_missing": review.pop()
    else: rows[0]["review_id"] = "../escape"
    with pytest.raises(ValueError): reconcile(plan, cases, rubrics, rows, review)


def receipt():
    row = {"review_id": "0" * 12, "text": "a", "condition": "marked", "tokens": 2,
           "model_calls": 2, "completion": "eos"}
    report = {"kind": "generation_trace", "identity": {}, "condition": "marked", "committed_token_ids": [0, 1],
              "model_calls": 2, "text": "a", "completion": "eos",
              "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
              "carrier_rendering": [{"pending_utf8_hex": ""}]}
    events = [{"phase": "start", "identity": {}, "condition": "marked", "prompt_token_ids": [9]}]
    for i, token in enumerate([0, 1]):
        events.extend([{"phase": "model_forward", "index": i},
            {"phase": "prepared", "raw_logits_sha256": "a" * 64, "weights_sha256": "b" * 64},
            {"phase": "commit_requested", "token_id": token, "draw": {"token_index": token,
                "integer_point": 0, "total_weight": 1, "transcript": []}},
            {"phase": "committed", "token_id": token}])
    events.append({"phase": "complete", "completion": "eos"})
    binding = SimpleNamespace(pieces=[b"a", None], eos_ids={1})
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [9], decode=lambda *a, **kw: "a")
    return row, report, events, binding, tokenizer, {"prompt": "source"}, {}, 2


def test_native_receipt_is_reconciled():
    assert audit_output(*receipt())["tokens"] == 2


@pytest.mark.parametrize("change", ["rewrite", "decoder", "prompt", "usage", "calls", "unknown_stop", "cap_eos", "early_eos", "extra_forward"])
def test_inference_receipt_mutations_fail(change):
    args = receipt()
    row, report, events, binding, tokenizer, *_ = args
    if change == "rewrite": row["text"] = report["text"] = "repaired"
    elif change == "decoder": tokenizer.decode = lambda *a, **kw: "other"
    elif change == "prompt": events[0]["prompt_token_ids"] = [8]
    elif change == "usage": report["usage"]["total_tokens"] = 4
    elif change == "calls": row["model_calls"] = report["model_calls"] = 3
    elif change in {"unknown_stop", "cap_eos"}:
        value = "unknown" if change == "unknown_stop" else "length"
        row["completion"] = report["completion"] = events[-1]["completion"] = value
    elif change == "early_eos": report["committed_token_ids"] = [1, 0]
    else: events.insert(-1, {"phase": "model_forward", "index": 2})
    with pytest.raises(ValueError): audit_output(*args)
