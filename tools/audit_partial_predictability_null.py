"""Verify surviving null receipts without repairing or approving an incomplete run.

No model inference, replacement controls, fitted thresholds or confidence bound.
The original complete-run audit stays failed and byte-for-byte unchanged.
"""
import argparse
import json
import math
from pathlib import Path

from audit_predictability_null import selected_event_bits, sha
from weighted_null import reference_tail


def reconcile(rows, summary, expected):
    if len(rows) != len(expected) or {r["source_index"] for r in rows} != set(expected):
        raise ValueError("Every planned attempt, including failures, must remain")
    failed, usable = [], []
    for row in rows:
        if "error" in row:
            if not row["error"] or any(k in row for k in ("flagged", "family_reference_tail", "weighted")):
                raise ValueError("Failed attempt cannot contain a detection verdict")
            failed.append(row["source_index"])
            continue
        if row.get("unavailable") is not False:
            raise ValueError("Unavailable selection is unsupported by this partial audit")
        tail = row["family_reference_tail"]
        if (type(tail) not in (float, int) or not math.isfinite(tail) or not 0 <= tail <= 1
                or type(row["flagged"]) is not bool or row["flagged"] != (tail <= .01)
                or type(row["baseline_flagged"]) is not bool):
            raise ValueError("Invalid usable-control verdict")
        usable.append(row)
    hits = sum(r["flagged"] for r in usable)
    baseline = sum(r["baseline_flagged"] for r in usable)
    if (not failed or not usable or summary["status"] != "incomplete"
            or summary["attempts"] != len(rows) or summary["errors"] != len(failed)
            or summary["unavailable"] != 0 or summary["fatal"] is not None
            or summary["hits"] != hits or summary["baseline_hits"] != baseline
            or summary["iid_only_upper_97_5"] is not None
            or summary["deployment_calibrated"] is not False):
        raise ValueError("Incomplete summary must preserve errors and withhold acceptance")
    transitions = {name: [] for name in ("both_flagged", "candidate_only", "baseline_only", "neither")}
    for row in usable:
        name = (("both_flagged" if row["baseline_flagged"] else "candidate_only") if row["flagged"]
                else ("baseline_only" if row["baseline_flagged"] else "neither"))
        transitions[name].append(row["source_index"])
    return usable, {"planned_attempts": len(rows), "usable_controls": len(usable),
                    "failed_indices": sorted(failed), "candidate_hits": hits,
                    "baseline_hits_same_usable_controls": baseline,
                    "transitions": {k: len(v) for k, v in transitions.items()},
                    "flagged_indices": {k: sorted(v) for k, v in transitions.items() if k != "neither"},
                    "iid_only_upper_97_5": None, "deployment_calibrated": False}


def audit(args):
    root = args.run
    plan = json.loads((root / "public/plan.json").read_text())
    summary = json.loads((root / "public/summary.json").read_text())
    original_audit_path = root / "public/integrity.json"
    original_audit = json.loads(original_audit_path.read_text())
    original_audit_sha = sha(original_audit_path)
    if (original_audit["status"] != "failed"
            or original_audit["summary_sha256"] != sha(root / "public/summary.json")
            or original_audit["results_sha256"] != sha(root / "public/results.jsonl")):
        raise ValueError("Preserved failed audit must bind to this exact run")
    paths = ((Path(__file__).with_name("develop_predictability_null.py"), "script_sha256"),
             (args.source, "source_sha256"),
             (args.development / "public/plan.json", "candidate_plan_sha256"),
             (args.development / "public/summary.json", "power_summary_sha256"),
             (args.null_study / "public/plan.json", "null_plan_sha256"),
             (args.null_study / "public/results.jsonl", "null_results_sha256"))
    for path, field in paths:
        if sha(path) != plan[field]:
            raise ValueError("Declared input changed: " + field)
    candidate = json.loads((args.development / "public/plan.json").read_text())
    for name, digest in candidate["dependencies_sha256"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("Frozen candidate changed")
    prior = json.loads((args.null_study / "public/plan.json").read_text())
    import hashlib
    keys = [(args.null_study / f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != prior["key_commitments"]:
        raise ValueError("Null keys changed")
    def lines(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    source = lines(args.source)
    originals_list = lines(args.null_study / "public/results.jsonl")
    originals = {r["source_index"]: r for r in originals_list}
    rows = lines(root / "public/results.jsonl")
    expected = {r["source_index"] for r in prior["records"]}
    if len(expected) != 500 or len(originals_list) != 500 or set(originals) != expected:
        raise ValueError("All 500 original controls required")
    usable, counts = reconcile(rows, summary, expected)
    for row in rows:
        original = originals[row["source_index"]]
        text = source[row["source_index"]]["response"]
        if (hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]
                or any(row[k] != original[k] for k in ("text_sha256", "words", "category"))):
            raise ValueError("Attempt source text or metadata changed")
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding = runtime_binding(max_steps=2048)
    checked, max_delta = 0, 0.
    for row in usable:
        text = source[row["source_index"]]["response"]
        path = root / (str(row["source_index"]) + ".selection.json")
        if sha(path) != row["selection_sha256"]:
            raise ValueError("Private selection hash differs")
        selection = json.loads(path.read_text())
        tails = []
        for index, key in enumerate(keys):
            bits = selected_event_bits(binding, text, key, selection)
            score = reference_tail(bits)
            delta = abs(score["reference_tail"] - reference_tail(bits, double_grid=True)["reference_tail"])
            max_delta = max(max_delta, delta)
            if (delta > 1e-10 or score["reference_tail"] != row["weighted"][index]["reference_tail"]
                    or score["centered_sum"] != row["weighted"][index]["centered_sum"]
                    or len(bits) != row["selected_events"]):
                raise ValueError("Selected-bit replay differs")
            tails.append(score["reference_tail"])
            checked += 1
        baseline = originals[row["source_index"]]["family_reference_tail"]["linear_10_to_1"] <= .01
        if row["family_reference_tail"] != min(1., 2 * min(tails)) or row["baseline_flagged"] != baseline:
            raise ValueError("Family score or original baseline differs")
        if checked % 100 == 0:
            print(json.dumps({"scores_replayed": checked}), flush=True)
    if sha(original_audit_path) != original_audit_sha:
        raise ValueError("Original failed audit changed during replay")
    return {"status": "partial_verified", "counts": counts, "scores_replayed": checked,
            "max_double_grid_difference": max_delta, "original_failed_audit_sha256": original_audit_sha,
            "summary_sha256": sha(root / "public/summary.json"),
            "results_sha256": sha(root / "public/results.jsonl"),
            "script_sha256": sha(Path(__file__)),
            "helper_sha256": sha(Path(__file__).with_name("audit_predictability_null.py")),
            "complete_run_accepted": False, "production_accepted": False,
            "scope": "Arithmetic integrity of surviving opened controls only; original run remains incomplete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "development", "null-study", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    destination = args.run / "public/partial-integrity.json"
    # Reserve before work, without touching the original complete-run audit.
    with destination.open("x") as stream:
        try:
            result = audit(args)
        except Exception as exc:
            result = {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
                      "complete_run_accepted": False, "production_accepted": False}
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "partial_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
