"""Independently reconcile all controls, private selections and public counts.

No model calls or new generation. A pass is receipt integrity, not a claim of
fresh validation, IID documents or an acceptable deployment false-positive rate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

from weighted_null import reference_tail


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_summary(rows, summary, expected_indices):
    total = len(expected_indices)
    if not total or len(rows)!=total or {r["source_index"] for r in rows}!=set(expected_indices):
        raise ValueError("Missing, duplicate or substituted controls")
    if any(r.get("error") or r.get("unavailable") is not False for r in rows):
        raise ValueError("Failed or unavailable controls cannot count as negatives")
    for row in rows:
        tail = row["family_reference_tail"]
        if (type(tail) not in (int,float) or not math.isfinite(tail) or not 0<=tail<=1
                or type(row["flagged"]) is not bool or row["flagged"]!=(tail<=.01)):
            raise ValueError("Reported flag differs from frozen threshold")
    hits = sum(r["flagged"] for r in rows)
    baseline = sum(r["baseline_flagged"] for r in rows)
    upper = float(beta.ppf(.975,hits+1,total-hits)) if hits<total else 1.
    if (summary["status"]!="completed" or summary["attempts"]!=total
            or summary["errors"]!=0 or summary["unavailable"]!=0 or summary["fatal"] is not None
            or summary["hits"]!=hits or summary["baseline_hits"]!=baseline
            or not math.isclose(summary["iid_only_upper_97_5"],upper,rel_tol=1e-10,abs_tol=1e-12)):
        raise ValueError("Public summary or IID-only bound does not reconcile")
    return {"controls":total,"hits":hits,"baseline_hits":baseline,"iid_only_upper_97_5":upper}


def selected_event_bits(binding, text, key, selection):
    """Pair replayed nonempty events with literal positions independently."""
    replay = binding.replay_text(text,key)
    entropies = selection["entropy_bits"]
    if (list(replay.token_ids)!=selection["token_ids"] or len(entropies)!=len(replay.token_ids)
            or hashlib.sha256(text.encode()).hexdigest()!=selection["text_sha256"]
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in entropies)):
        raise ValueError("Private selection text, tokens or entropy differs")
    positions = [i for i,token in enumerate(replay.token_ids) if binding.profile.classes[token] is not None]
    pairs = list(zip(positions,replay.events,strict=True))
    eligible = [i for i,event in pairs if event.eligible]
    selected = [i for i,event in pairs if event.eligible and entropies[i]>=.5]
    if eligible!=selection["eligible_positions"] or selected!=selection["selected_positions"]:
        raise ValueError("Private mask differs from fixed key-blind selection")
    chosen = set(selected)
    return np.asarray([event.bits for i,event in pairs if i in chosen],dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run","development","null-study","source"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    root = args.run
    summary = json.loads((root/"public/summary.json").read_text())
    if (root/"public/integrity.json").exists():
        raise ValueError("Never overwrite an existing audit")
    error, checked, max_delta, counts = None, 0, 0., None
    try:
        plan = json.loads((root/"public/plan.json").read_text())
        if sha(Path(__file__).with_name("develop_predictability_null.py"))!=plan["script_sha256"]:
            raise ValueError("Null driver changed")
        for path,field in ((args.source,"source_sha256"),(args.development/"public/plan.json","candidate_plan_sha256"),
                           (args.development/"public/summary.json","power_summary_sha256"),
                           (args.null_study/"public/plan.json","null_plan_sha256"),
                           (args.null_study/"public/results.jsonl","null_results_sha256")):
            if sha(path)!=plan[field]:
                raise ValueError("Declared input changed: "+field)
        candidate = json.loads((args.development/"public/plan.json").read_text())
        for name,h in candidate["dependencies_sha256"].items():
            if sha(Path(__file__).with_name(name))!=h:
                raise ValueError("Frozen candidate changed")
        prior = json.loads((args.null_study/"public/plan.json").read_text())
        keys = [(args.null_study/f"owner-{i}.key").read_bytes() for i in range(2)]
        if [hashlib.sha256(k).hexdigest() for k in keys]!=prior["key_commitments"]:
            raise ValueError("Null keys changed")
        corpus = [json.loads(line) for line in args.source.read_text().splitlines()]
        originals = {r["source_index"]:r for r in [json.loads(line) for line in (args.null_study/"public/results.jsonl").read_text().splitlines()]}
        rows = [json.loads(line) for line in (root/"public/results.jsonl").read_text().splitlines()]
        expected = {r["source_index"] for r in prior["records"]}
        if len(expected)!=500 or set(originals)!=expected:
            raise ValueError("All 500 original controls required")
        counts = validate_summary(rows,summary,expected)
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        binding = runtime_binding(max_steps=2048)
        for row in rows:
            original = originals[row["source_index"]]
            text = corpus[row["source_index"]]["response"]
            if (hashlib.sha256(text.encode()).hexdigest()!=row["text_sha256"]
                    or any(row[k]!=original[k] for k in ("text_sha256","words","category"))):
                raise ValueError("Original control text or metadata changed")
            path = root/(str(row["source_index"])+".selection.json")
            if sha(path)!=row["selection_sha256"]:
                raise ValueError("Private selection hash differs")
            selection = json.loads(path.read_text())
            tails = []
            for i,key in enumerate(keys):
                bits = selected_event_bits(binding,text,key,selection)
                value = reference_tail(bits)
                delta = abs(value["reference_tail"]-reference_tail(bits,double_grid=True)["reference_tail"])
                max_delta = max(max_delta,delta)
                if (delta>1e-10 or value["reference_tail"]!=row["weighted"][i]["reference_tail"]
                        or value["centered_sum"]!=row["weighted"][i]["centered_sum"]
                        or len(bits)!=row["selected_events"]):
                    raise ValueError("Selected-bit reference replay differs")
                tails.append(value["reference_tail"])
                checked += 1
            if (row["family_reference_tail"]!=min(1.,2*min(tails))
                    or row["baseline_flagged"]!=(original["family_reference_tail"]["linear_10_to_1"]<=.01)):
                raise ValueError("Family tail or original baseline differs")
    except Exception as exc:
        error = {"type":type(exc).__name__,"message":str(exc)}
    result = {"status":"pass" if error is None and checked==1000 else "failed", "error":error,
              "scores_replayed":checked,"max_double_grid_difference":max_delta,"counts":counts,
              "script_sha256":sha(Path(__file__)),"summary_sha256":sha(root/"public/summary.json"),
              "results_sha256":sha(root/"public/results.jsonl"),
              "scope":"Receipt and arithmetic integrity only; opened controls and IID-only bound are not deployment qualification"}
    (root/"public/integrity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return 0 if result["status"]=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
