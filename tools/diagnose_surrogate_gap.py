"""Compare retained surrogate and original-prompt likelihoods on identical positions.

Oracle diagnosis only: original generation probabilities are unavailable to a
pasted-text detector. This does not change a scorer, threshold or generation.
Use every short marked response from the already-opened twelve-pair study.
"""
import argparse
import json
import math
from pathlib import Path

from develop_residual_bet import sha, write


def half_mixture(log_ratio):
    if not math.isfinite(log_ratio):
        raise ValueError("Finite retained log ratio required")
    return max(0., log_ratio) + math.log1p(math.exp(-abs(log_ratio))) - math.log(2.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "surrogate", "oracle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source_plan = args.study / "public/plan.json"
    old_plan = json.loads((args.oracle / "public/plan.json").read_text())
    old_summary = json.loads((args.oracle / "public/summary.json").read_text())
    old_audit = json.loads((args.oracle / "public/integrity.json").read_text())
    surrogate_plan = json.loads((args.surrogate / "public/plan.json").read_text())
    assert old_plan["study_plan_sha256"] == surrogate_plan["source_plan_sha256"] == sha(source_plan)
    assert old_audit["status"] == "pass" and old_audit["all_step_aggregates_recomputed"]
    assert old_audit["summary_sha256"] == sha(args.oracle / "public/summary.json")
    assert surrogate_plan["mixture"] == .5 and surrogate_plan["fixed_log_cutoff"] == math.log(200.)
    sources = [json.loads(p.read_text()) for p in sorted((args.study / "public").glob("weighted-*.json"))]
    selected = [r for r in sources if r["condition"] == "marked" and 100 <= r["words"] <= 400]
    assert len(sources) == 24 and len(selected) == 5
    assert set(old_plan["cases"]) == {r["id"] for r in selected}
    source_hashes = {}
    for row in selected:
        name = row["id"]
        for suffix, key in (("report.json", "source_reports"), ("journal.jsonl", "source_journals")):
            assert sha(args.study / name / suffix) == old_plan[key][name]
        source = args.study / "public" / (name + ".json")
        assert sha(source) == old_plan["source_public_rows"][name] == surrogate_plan["source_rows_sha256"][source.name]
        source_hashes[name] = {str(p): sha(p) for p in (
            args.oracle / (name + ".steps.jsonl"), args.surrogate / (name + ".scores.json"),
            args.surrogate / (name + ".heads.json"), args.surrogate / "public" / (name + ".json"))}
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    write(public / "plan.json", {"scope": __doc__, "script_sha256": sha(Path(__file__)),
        "source_plan_sha256": sha(source_plan), "source_hashes": source_hashes,
        "oracle_audit_sha256": sha(args.oracle / "public/integrity.json"),
        "oracle_plan_sha256": sha(args.oracle / "public/plan.json"),
        "surrogate_plan_sha256": sha(args.surrogate / "public/plan.json"),
        "cases": [r["id"] for r in selected], "mixture": .5, "fixed_log_cutoff": math.log(200.),
        "alignment": "Exact literal tokens must equal the complete visible prefix of generation tokens; same surrogate-scored positions in both arms",
        "fresh_confirmation": False, "new_inference": False, "detector_promotion": False})
    results = []
    for row in selected:
        name = row["id"]
        steps = [json.loads(line) for line in (args.oracle / (name + ".steps.jsonl")).read_text().splitlines()]
        previous = next(r for r in old_summary["results"] if r["id"] == name)
        assert previous["all_raw_logits_distributions_tokens_draws_match"] and previous["steps"] == len(steps)
        assert math.isclose(math.fsum(s["selected_log2_ratio"] for s in steps),
                            previous["sums"]["selected_log2_ratio"], rel_tol=0, abs_tol=1e-10)
        report = json.loads((args.study / name / "report.json").read_text())["report"]
        generated = report["payload"]["committed_token_ids"]
        assert [s["index"] for s in steps] == list(range(len(steps)))
        assert [s["token_id"] for s in steps] == generated
        scores_path = args.surrogate / (name + ".scores.json")
        heads_path = args.surrogate / (name + ".heads.json")
        old_row = json.loads((args.surrogate / "public" / (name + ".json")).read_text())
        assert sha(scores_path) == old_row["scores_sha256"] and sha(heads_path) == old_row["heads_sha256"]
        score = json.loads(scores_path.read_text())[row["key_index"]]
        heads = json.loads(heads_path.read_text())
        ids = heads["token_ids"]
        assert ids == generated[:len(ids)] and len(generated) - len(ids) in (0, 1)
        assert len(score["terms"]) == len(ids)
        eligible = [t["index"] for t in score["terms"] if t["reason"] == "scored"]
        assert all(t["index"] == i and t["token"] == ids[i] for i, t in enumerate(score["terms"]))
        assert all(steps[i]["fresh_context"] for i in eligible)
        original_terms = [half_mixture(steps[i]["selected_log2_ratio"] * math.log(2.)) for i in eligible]
        surrogate_terms = [score["terms"][i]["log_ratio"] for i in eligible]
        original_value, surrogate_value = math.fsum(original_terms), math.fsum(surrogate_terms)
        assert math.isclose(surrogate_value, score["working_log_ratio"], rel_tol=0, abs_tol=1e-10)
        result = {"id": name, "words": row["words"], "identical_scored_positions": len(eligible),
            "oracle_log_evidence": original_value, "surrogate_log_evidence": surrogate_value,
            "oracle_above_reference": original_value >= math.log(200.),
            "surrogate_above_reference": surrogate_value >= math.log(200.),
            "original_prompt_required": True, "deployment_calibrated": False}
        write(public / (name + ".json"), result)
        results.append(result)
    summary = {"status": "pass", "cases": len(results),
        "aligned_positions": sum(r["identical_scored_positions"] for r in results),
        "oracle_above_reference": sum(r["oracle_above_reference"] for r in results),
        "surrogate_above_reference": sum(r["surrogate_above_reference"] for r in results),
        "scope": __doc__, "new_model_inference": False, "detector_promotion": False,
        "results": results}
    write(public / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
