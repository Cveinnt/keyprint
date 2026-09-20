"""Independent token-mass replay of every prompt-conditioned likelihood term.

Use Python math and direct per-label PRF calls, not the scorer's NumPy/SciPy
or grouped bit-table transform. Audit retained model outputs, identities,
causal shared-prefix heads and counts; this does not re-run model kernels or
establish held-out detection power, calibration or a prompt-free detector.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logsum(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def independent_ratio(profile, key, context, support, probabilities, token):
    active = [(t, p) for t, p in zip(support, probabilities, strict=True) if profile.classes[t] is not None]
    labels = {profile.classes[t] for t, _ in active}
    if len(labels) < 2:
        return 0.
    bits = {label: profile.bits(key, context, label) for label in labels}
    values = [math.log(p) for _, p in active]
    total = logsum(values)
    weights = [v-total for v in values]
    index = [t for t, _ in active].index(token)
    initial = weights[index]
    for layer in range(profile.config.layers):
        flags = [bits[profile.classes[t]][layer] for t, _ in active]
        if all(flags) or not any(flags):
            continue
        # Direct token masses: zero-bit tokens survive only against a zero;
        # one-bit tokens survive against either bit, with double-count removed.
        zero = logsum([v for v, flag in zip(weights, flags, strict=True) if not flag])
        total = logsum(weights)
        one_factor = logsum([zero, total])
        updated = [v+(one_factor if flag else zero) for v, flag in zip(weights, flags, strict=True)]
        normalizer = logsum(updated)
        weights = [v-normalizer for v in updated]
    return weights[index] - initial


def half_factor(ratio):
    return max(0., ratio) + math.log1p(math.exp(-abs(ratio))) - math.log(2.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "development", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from transformers import AutoTokenizer
    from keyprint.backends.mlx import ASSETS, verify_assets
    from keyprint import sampling
    verify_assets(args.model)
    binding = runtime_binding(max_steps=2048)
    profile = binding.profile
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False, local_files_only=True)
    public = args.development / "public"
    plan = json.loads((public / "plan.json").read_text())
    identity = json.loads((public / "identity.json").read_text())
    assert identity == {"model_assets":ASSETS,"profile":profile.identity_receipt(),"sampling":sampling.identity()}
    assert plan["script_sha256"] == sha(Path(__file__).with_name("develop_prompt_conditioned_likelihood.py"))
    for name, digest in plan["dependencies_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(args.study / "public/plan.json") == plan["source_plan_sha256"]
    source_plan = json.loads((args.study / "public/plan.json").read_text())
    tasks = {t["source_index"]:t for t in source_plan["tasks"]}
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    assert [hashlib.sha256(k).hexdigest() for k in keys] == source_plan["key_commitments"]
    assert plan["fixed_log_cutoff"] == math.log(200.) and plan["mixture"] == .5
    totals = {"heads":0,"scores":0,"terms":0,"independent_transform_terms":0,"shared_prefix_heads":0}
    rows, paired = [], {}
    maximum_term_error, maximum_score_error = 0., 0.
    for filename, digest in plan["source_rows_sha256"].items():
        source_path = args.study / "public" / filename
        assert sha(source_path) == digest
        source = json.loads(source_path.read_text())
        row = json.loads((public / filename).read_text())
        assert all(row[k] == source[k] for k in ("id","condition","key_index","words","completion"))
        head_path = args.development / (source["id"] + ".heads.json")
        score_path = args.development / (source["id"] + ".scores.json")
        assert sha(head_path) == row["heads_sha256"] and sha(score_path) == row["scores_sha256"]
        data = json.loads(head_path.read_text())
        scores = json.loads(score_path.read_text())
        prompt = tasks[source["source_index"]]["prompt"]
        prefix = tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert data["conditioning_prefix_ids"] == prefix
        assert data["original_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert data["original_prompt_used"] and not data["private_generation_data_used"] and not data["key_used_for_model_inference"]
        assert data["token_ids"] == list(binding.encode_visible(source["text"]))
        assert data["text_sha256"] == hashlib.sha256(source["text"].encode()).hexdigest()
        assert len(scores) == 2 and len(data["heads"]) == len(data["token_ids"])
        paired.setdefault(source["source_index"],{})[source["condition"]] = data
        totals["heads"] += len(data["heads"])
        for key, score in zip(keys, scores, strict=True):
            context, seen, calculated = (), set(), []
            scored, outside = 0, 0
            for i, (token, head, term) in enumerate(zip(data["token_ids"],data["heads"],score["terms"],strict=True)):
                support, probabilities = head["support"], head["probabilities"]
                assert support == sorted(set(support)) and len(support) == len(probabilities) <= 100
                assert all(math.isfinite(p) and p > 0 for p in probabilities) and abs(math.fsum(probabilities)-1.) < 1e-12
                assert term["index"] == i and term["token"] == token
                label = profile.classes[token]
                reason = "excluded_label" if label is None else "repeated_context" if context in seen else "outside_surrogate_support" if token not in support else "scored"
                assert term["reason"] == reason
                base = probabilities[support.index(token)] if token in support else 0.
                assert term["base"] == base
                contribution = 0.
                if reason == "scored":
                    ratio = independent_ratio(profile,key,context,support,probabilities,token)
                    contribution = half_factor(ratio)
                    assert abs(term["marked_log_probability"] - math.log(base) - ratio) < 1e-10
                    scored += 1
                    totals["independent_transform_terms"] += 1
                else:
                    assert term["marked_log_probability"] is None
                outside += reason == "outside_surrogate_support"
                error = abs(contribution-term["log_ratio"])
                maximum_term_error = max(maximum_term_error,error)
                assert error < 1e-10
                calculated.append(contribution)
                if label is not None:
                    seen.add(context)
                    context = (*context,label)[-profile.config.history:]
                totals["terms"] += 1
            value = math.fsum(calculated)
            maximum_score_error = max(maximum_score_error,abs(value-score["working_log_ratio"]))
            assert abs(value-score["working_log_ratio"]) < 1e-7
            assert score["flagged"] == (value >= math.log(200.))
            assert score["scored_events"] == scored and score["outside_support"] == outside
            assert score["calibrated"] is False
            totals["scores"] += 1
        assert row["working_log_ratios"] == [s["working_log_ratio"] for s in scores]
        assert row["matching_flagged"] == scores[row["key_index"]]["flagged"]
        assert row["other_flagged"] == scores[1-row["key_index"]]["flagged"]
        rows.append(row)
        print(json.dumps({"audited":source["id"],"terms":totals["terms"]}),flush=True)
    assert len(rows) == 24 and len(paired) == 12
    for pair in paired.values():
        left,right=pair["ordinary"],pair["marked"]
        # At the first different observed token, its predictive input prefix
        # is still identical. Later tokens must not affect that retained head.
        for i,(a,b) in enumerate(zip(left["token_ids"],right["token_ids"])):
            assert left["heads"][i]["raw_head_sha256"] == right["heads"][i]["raw_head_sha256"]
            totals["shared_prefix_heads"] += 1
            if a != b:
                break
    groups = {c:{f:sum(r[f] for r in rows if r["condition"]==c) for f in ("matching_flagged","other_flagged")} for c in ("ordinary","marked")}
    short = [r for r in rows if r["condition"]=="marked" and 100<=r["words"]<=400]
    short_hits=sum(r["matching_flagged"] for r in short)
    passed = groups["marked"]["matching_flagged"]>=10 and short_hits>=4 and not any(groups[c][f] for c,f in (("ordinary","matching_flagged"),("ordinary","other_flagged"),("marked","other_flagged")))
    summary=json.loads((public/"summary.json").read_text())
    assert summary["status"]=="completed" and summary["attempts"]==24 and summary["errors"]==0 and summary["fatal"] is None
    assert summary["groups"]==groups and summary["short_marked"]=={"count":5,"hits":short_hits} and summary["power_gate_passed"]==passed
    result={"status":"pass","scope":__doc__,"totals":totals,"maximum_term_error":maximum_term_error,
        "maximum_score_error":maximum_score_error,"all_score_terms_recomputed":True,"model_kernels_rerun":False,
        "plan_sha256":sha(public/"plan.json"),"summary_sha256":sha(public/"summary.json"),
        "audit_source_sha256":sha(Path(__file__)),"sdk_promotion":False,"deployment_calibrated":False}
    with (public/"integrity.json").open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
