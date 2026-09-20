"""Freeze one weighted residual score on all 24 already-opened source responses.

Reuses the archived, bounded-audited residuals. Does not rerun their underlying
model heads or certify every residual. No generation, coefficient fitting,
threshold search, input replacement or SDK detector promotion.
"""
import argparse
import json
from pathlib import Path

from develop_residual_bet import sha, write
from layer_weighted_bet import aggregate, weights, STAKES, LOG_CUTOFF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "residuals", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.study / "public/plan.json").read_text())
    old_public = args.residuals / "public"
    old_plan = json.loads((old_public / "plan.json").read_text())
    old_summary = json.loads((old_public / "summary.json").read_text())
    audit = json.loads((old_public / "diagnostic-audit.json").read_text())
    if (audit["status"] != "pass" or audit["plan_sha256"] != sha(old_public / "plan.json")
            or audit["summary_sha256"] != sha(old_public / "summary.json")
            or old_summary["status"] != "completed" or old_summary["attempts"] != 24
            or old_summary["errors"] != 0 or old_plan["source_plan_sha256"] != sha(args.study / "public/plan.json")
            or old_plan["stakes"] != list(STAKES) or old_plan["fixed_log_cutoff"] != LOG_CUTOFF):
        raise ValueError("Complete unchanged residual archive and diagnostic audit required")
    expected = {(task["source_index"], condition) for task in original["tasks"] for condition in ("ordinary", "marked")}
    sources = sorted((args.study / "public").glob("weighted-*.json"))
    source_rows = [json.loads(p.read_text()) for p in sources]
    if (len(sources) != 24 or len(expected) != 24
            or {(r["source_index"], r["condition"]) for r in source_rows} != expected
            or {p.name: sha(p) for p in sources} != old_plan["source_rows_sha256"]):
        raise ValueError("Every original source response required")
    retained = []
    hashes = {}
    for path, source in zip(sources, source_rows, strict=True):
        row_path = old_public / path.name
        row = json.loads(row_path.read_text())
        detail = args.residuals / (source["id"] + ".scores.json")
        scores = json.loads(detail.read_text())
        if (row.get("error") or sha(detail) != row["scores_sha256"] or len(scores) != 2
                or source["id"] != path.stem
                or any(row[k] != source[k] for k in ("id", "condition", "key_index", "words", "completion"))):
            raise ValueError("Retained residual identity differs")
        retained.append((source, scores))
        hashes[path.name] = {"public_row": sha(row_path), "scores": sha(detail)}
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    declaration = {"scope": __doc__, "script_sha256": sha(Path(__file__)),
        "dependencies_sha256": {name: sha(Path(__file__).with_name(name)) for name in
            ("layer_weighted_bet.py", "develop_residual_bet.py", "residual_bet.py")},
        "source_plan_sha256": sha(args.study / "public/plan.json"),
        "source_rows_sha256": old_plan["source_rows_sha256"],
        "residual_plan_sha256": sha(old_public / "plan.json"),
        "residual_summary_sha256": sha(old_public / "summary.json"),
        "diagnostic_audit_sha256": sha(old_public / "diagnostic-audit.json"),
        "retained_files_sha256": hashes, "layer_weights": weights(30), "stakes": STAKES,
        "factor": "1 + stake * fixed_layer_weight * archived_residual",
        "aggregation": "multiply all eligible layers/positions per stake, then uniform whole-document mixture",
        "fixed_log_cutoff": LOG_CUTOFF, "comparison": "inclusive",
        "gate": "At least 10/12 marked and 4/5 short marked, zero ordinary/wrong-key flags; full residual replay and fresh confirmation required before promotion",
        "reference": "https://github.com/google-deepmind/synthid-text/blob/addb4a158143c7c6851a1308f78b89fceed59683/src/synthid_text/detector_mean.py",
        "reference_scope": "Relative layer weighting only; independent betting calculation, not a novelty claim",
        "fresh_data": False, "new_model_inference": False, "full_residual_replay": False}
    # Declare before any candidate score is evaluated.
    write(public / "plan.json", declaration)
    rows = []
    for source, archived in retained:
        row = {k: source[k] for k in ("id", "condition", "key_index", "words", "completion")}
        try:
            values = [aggregate(value["terms"]) for value in archived]
            row.update(scores=values, matching_flagged=values[source["key_index"]]["flagged"],
                       other_flagged=values[1 - source["key_index"]]["flagged"])
        except (ValueError, KeyError, TypeError) as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write(public / (source["id"] + ".json"), row)
        rows.append(row)
    errors = sum("error" in row for row in rows)
    groups = {condition: {field: sum(row.get(field, False) for row in rows if row["condition"] == condition)
        for field in ("matching_flagged", "other_flagged")} for condition in ("ordinary", "marked")}
    short = [row for row in rows if row["condition"] == "marked" and 100 <= row["words"] <= 400]
    hits = sum(row.get("matching_flagged", False) for row in short)
    passed = (not errors and len(short) == 5 and groups["marked"]["matching_flagged"] >= 10 and hits >= 4
              and not groups["ordinary"]["matching_flagged"] and not groups["ordinary"]["other_flagged"]
              and not groups["marked"]["other_flagged"])
    summary = {"status": "incomplete" if errors else "completed", "attempts": len(rows), "errors": errors,
        "groups": groups, "short_marked": {"count": len(short), "hits": hits},
        "power_gate_passed": bool(passed), "full_residual_replay": False,
        "deployment_calibrated": False, "sdk_promotion": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
