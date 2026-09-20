"""Independent aggregation audit, not full model-head or residual replay.

Check all retained source hashes and recompute weighted products per token,
then sum their logs. The candidate instead sums log1p factors across all layers.
No candidate scoring function is called. Every source response is required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "residuals", "development"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    public = args.development / "public"
    plan = json.loads((public / "plan.json").read_text())
    assert plan["script_sha256"] == sha(Path(__file__).with_name("develop_layer_weighted_bet.py"))
    for name, digest in plan["dependencies_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(args.study / "public/plan.json") == plan["source_plan_sha256"]
    old_public = args.residuals / "public"
    for name, field in (("plan.json", "residual_plan_sha256"), ("summary.json", "residual_summary_sha256"),
                        ("diagnostic-audit.json", "diagnostic_audit_sha256")):
        assert sha(old_public / name) == plan[field]
    original = json.loads((args.study / "public/plan.json").read_text())
    expected = {(t["source_index"], c) for t in original["tasks"] for c in ("ordinary", "marked")}
    source_files = sorted((args.study / "public").glob("weighted-*.json"))
    assert {p.name: sha(p) for p in source_files} == plan["source_rows_sha256"]
    assert {p.name for p in public.glob("weighted-*.json")} == set(plan["source_rows_sha256"])
    assert len(expected) == len(source_files) == 24
    assert plan["stakes"] == [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]
    weights = [(290 - 9 * layer) / 290 for layer in range(30)]
    assert plan["layer_weights"] == weights
    cutoff = math.log(200.)
    assert plan["fixed_log_cutoff"] == cutoff and plan["comparison"] == "inclusive"
    rows, seen = [], set()
    scores_count, terms_count, max_delta = 0, 0, 0.
    for path in source_files:
        source = json.loads(path.read_text())
        seen.add((source["source_index"], source["condition"]))
        result = json.loads((public / path.name).read_text())
        assert "error" not in result
        assert all(result[k] == source[k] for k in ("id", "condition", "key_index", "words", "completion"))
        assert sha(old_public / path.name) == plan["retained_files_sha256"][path.name]["public_row"]
        saved = args.residuals / (source["id"] + ".scores.json")
        assert sha(saved) == plan["retained_files_sha256"][path.name]["scores"]
        archived = json.loads(saved.read_text())
        assert len(archived) == len(result["scores"]) == 2
        flags = []
        for old, score in zip(archived, result["scores"], strict=True):
            logs = [[] for _ in plan["stakes"]]
            for index, term in enumerate(old["terms"]):
                assert term["index"] == index
                residuals = term["residuals"]
                assert term["reason"] in ("scored", "excluded_label", "repeated_context", "outside_surrogate_support")
                assert len(residuals) == (30 if term["reason"] == "scored" else 0)
                assert all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1 for v in residuals)
                for stake, component in zip(plan["stakes"], logs, strict=True):
                    product = math.prod(1 + stake * w * r for w, r in zip(weights, residuals))
                    assert product > 0 and math.isfinite(product)
                    component.append(math.log(product))
                terms_count += 1
            components = [math.fsum(v) for v in logs]
            # These finite development scores fit directly in binary64 exp.
            evidence = math.log(math.fsum(math.exp(v) for v in components) / 5)
            assert len(score["component_log_evidence"]) == 5
            deltas = [abs(a - b) for a, b in zip(components, score["component_log_evidence"], strict=True)]
            deltas.append(abs(evidence - score["working_log_evidence"]))
            assert max(deltas) < 1e-10
            max_delta = max(max_delta, *deltas)
            assert score["calibrated"] is False and score["flagged"] == (evidence >= cutoff)
            flags.append(score["flagged"])
            scores_count += 1
        assert result["matching_flagged"] == flags[source["key_index"]]
        assert result["other_flagged"] == flags[1 - source["key_index"]]
        rows.append(result)
    assert seen == expected
    groups = {condition: {field: sum(row[field] for row in rows if row["condition"] == condition)
        for field in ("matching_flagged", "other_flagged")} for condition in ("ordinary", "marked")}
    short = [r for r in rows if r["condition"] == "marked" and 100 <= r["words"] <= 400]
    short_hits = sum(row["matching_flagged"] for row in short)
    passed = (groups["marked"]["matching_flagged"] >= 10 and len(short) == 5 and short_hits >= 4
        and not groups["ordinary"]["matching_flagged"] and not groups["ordinary"]["other_flagged"]
        and not groups["marked"]["other_flagged"])
    summary = json.loads((public / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["attempts"] == 24 and summary["errors"] == 0
    assert summary["groups"] == groups and summary["short_marked"] == {"count": len(short), "hits": short_hits}
    assert summary["power_gate_passed"] == passed
    assert summary["deployment_calibrated"] is False and summary["sdk_promotion"] is False
    result = {"status": "pass", "scope": __doc__, "scores": scores_count, "terms": terms_count,
        "maximum_log_difference": max_delta, "plan_sha256": sha(public / "plan.json"),
        "summary_sha256": sha(public / "summary.json"), "auditor_sha256": sha(Path(__file__)),
        "full_residual_replay": False, "eligible_for_null_expansion": False, "sdk_promotion": False}
    with (public / "aggregation-audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
