"""Audit a failed development screen without claiming full residual replay.

Verify every source/head/score hash, token identity, eligibility decision, stake
aggregation and count. Independently recompute residuals at each document's
first, quarter, middle, three-quarter and last scored event using Profile.bits
and token-level masses rather than the candidate's grouped bit-table routine.
This bounded diagnostic audit never authorizes null expansion or promotion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "study", "heads", "prior-heads"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from keyprint._engine.legacy._impl.research import token_source_sparse_execution
    binding = runtime_binding(max_steps=2048)
    profile = binding.profile
    public = args.development / "public"
    plan = json.loads((public / "plan.json").read_text())
    assert sha(Path(__file__).with_name("develop_residual_bet.py")) == plan["script_sha256"]
    for name, digest in plan["dependencies_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(Path(token_source_sparse_execution.__file__)) == plan["bit_source_sha256"]
    assert sha(args.study / "public/plan.json") == plan["source_plan_sha256"]
    original = json.loads((args.study / "public/plan.json").read_text())
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    assert [hashlib.sha256(k).hexdigest() for k in keys] == original["key_commitments"]
    assert sha(args.heads / "public/plan.json") == plan["archived_head_plan_sha256"]
    assert sha(args.heads / "public/identity.json") == plan["archived_head_identity_sha256"]
    rows, totals = [], {"heads": 0, "scores": 0, "terms": 0, "independent_residual_events": 0}
    for filename, digest in plan["source_rows_sha256"].items():
        path = args.study / "public" / filename
        assert sha(path) == digest
        source = json.loads(path.read_text())
        row = json.loads((public / filename).read_text())
        head_path = args.heads / (source["id"] + ".heads.json")
        assert sha(head_path) == row["heads_sha256"] == plan["heads_sha256"][head_path.name]
        heads = json.loads(head_path.read_text())
        prior = json.loads((args.prior_heads / (source["id"] + ".selection.json")).read_text())
        assert heads["token_ids"] == list(binding.encode_visible(source["text"])) == prior["token_ids"]
        assert heads["text_sha256"] == hashlib.sha256(source["text"].encode()).hexdigest() == prior["text_sha256"]
        assert heads["fixed_prefix_ids"] == prior["fixed_prefix_ids"]
        assert [h["raw_head_sha256"] for h in heads["heads"]] == prior["raw_head_sha256"]
        totals["heads"] += len(heads["heads"])
        detail = args.development / (source["id"] + ".scores.json")
        assert sha(detail) == row["scores_sha256"]
        scores = json.loads(detail.read_text())
        assert len(scores) == 2
        for key, score in zip(keys, scores, strict=True):
            assert len(score["terms"]) == len(heads["token_ids"])
            eligible = [i for i, t in enumerate(score["terms"]) if t["reason"] == "scored"]
            sample = {eligible[(len(eligible) - 1) * j // 4] for j in range(5)} if eligible else set()
            context, seen = (), set()
            for i, (token, head, term) in enumerate(zip(heads["token_ids"], heads["heads"], score["terms"], strict=True)):
                assert term["index"] == i and term["token"] == token
                support = head["support"]
                values = head["probabilities"]
                assert support == sorted(set(support)) and len(support) == len(values) <= 100
                assert all(math.isfinite(p) and p > 0 for p in values) and abs(math.fsum(values)-1) < 1e-12
                label = profile.classes[token]
                reason = ("excluded_label" if label is None else "repeated_context" if context in seen
                          else "outside_surrogate_support" if token not in support else "scored")
                assert term["reason"] == reason
                assert len(term["residuals"]) == (profile.config.layers if reason == "scored" else 0)
                assert all(math.isfinite(v) and -1 <= v <= 1 for v in term["residuals"])
                if i in sample:
                    active = [(t, p) for t, p in zip(support, values, strict=True) if profile.classes[t] is not None]
                    mass = math.fsum(p for _, p in active)
                    bits = {profile.classes[t]: profile.bits(key, context, profile.classes[t]) for t, _ in active}
                    for layer, residual in enumerate(term["residuals"]):
                        expectation = math.fsum(p * bits[profile.classes[t]][layer] for t, p in active) / mass
                        reference = bits[label][layer] - expectation
                        assert abs(reference - residual) < 2e-14
                    totals["independent_residual_events"] += 1
                if label is not None:
                    seen.add(context)
                    context = (*context, label)[-profile.config.history:]
                totals["terms"] += 1
            component = [math.fsum(math.log1p(stake * value) for t in score["terms"] for value in t["residuals"])
                         for stake in plan["stakes"]]
            assert component == score["component_log_evidence"]
            maximum = max(component)
            value = maximum + math.log(math.fsum(math.exp(x-maximum) for x in component)/len(component))
            assert abs(value-score["working_log_evidence"]) < 1e-11
            assert score["flagged"] == (value >= plan["fixed_log_cutoff"])
            assert score["scored_events"] == len(eligible)
            totals["scores"] += 1
        assert row["working_log_evidence"] == [s["working_log_evidence"] for s in scores]
        assert row["matching_flagged"] == scores[row["key_index"]]["flagged"]
        assert row["other_flagged"] == scores[1-row["key_index"]]["flagged"]
        rows.append(row)
    assert len(rows) == 24
    groups = {c: {field: sum(r[field] for r in rows if r["condition"] == c)
        for field in ("matching_flagged", "other_flagged")} for c in ("ordinary", "marked")}
    short = [r for r in rows if r["condition"] == "marked" and 100 <= r["words"] <= 400]
    summary = json.loads((public / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["errors"] == 0
    assert summary["groups"] == groups
    assert summary["short_marked"] == {"count": len(short), "hits": sum(r["matching_flagged"] for r in short)}
    assert summary["power_gate_passed"] is False
    result = {"status": "pass", "scope": __doc__, "totals": totals,
        "plan_sha256": sha(public / "plan.json"), "summary_sha256": sha(public / "summary.json"),
        "audit_source_sha256": sha(Path(__file__)), "full_residual_replay": False,
        "eligible_for_null_expansion": False, "sdk_promotion": False}
    with (public / "diagnostic-audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
