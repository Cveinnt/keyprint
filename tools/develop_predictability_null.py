"""Evaluate the frozen predictability candidate on 500 opened null controls.

Requires the declared power-screen gate to pass. No fitting or new generation.
The output is development evidence, not fresh fixed-key deployment validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from predictability_filter import measure, selected_bits
from weighted_null import reference_tail, two_key_tail
from validate_null_corpus import SOURCE_SHA256, iid_binomial_upper


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "source", "null-study", "model", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    power_plan = json.loads((args.development/"public/plan.json").read_text())
    power = json.loads((args.development/"public/summary.json").read_text())
    if power["status"] != "completed" or power["expand_to_null_controls"] is not True:
        raise ValueError("Frozen power-screen gate did not pass")
    for name, expected in power_plan["dependencies_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Frozen candidate source differs: "+name)
    integrity = json.loads((args.development/"public/integrity.json").read_text())
    if integrity["status"] != "pass" or integrity["candidate_hashes"] != power_plan["dependencies_sha256"]:
        raise ValueError("Candidate integrity and prefix stability must pass")
    if (integrity["candidate_plan_sha256"]!=sha(args.development/"public/plan.json")
            or integrity["power_summary_sha256"]!=sha(args.development/"public/summary.json")):
        raise ValueError("Candidate audit belongs to different power evidence")
    if sha(args.source) != SOURCE_SHA256:
        raise ValueError("Pinned source differs")
    plan_path = args.null_study/"public/plan.json"
    prior = json.loads(plan_path.read_text())
    source_rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    originals = [json.loads(line) for line in (args.null_study/"public/results.jsonl").read_text().splitlines()]
    if len(originals)!=500 or len({r["source_index"] for r in originals})!=500:
        raise ValueError("All 500 distinct original controls required")
    if {r["source_index"] for r in originals}!={r["source_index"] for r in prior["records"]}:
        raise ValueError("Original null selection differs")
    keys = [(args.null_study/f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != prior["key_commitments"]:
        raise ValueError("Original null keys differ")
    texts = []
    for row in originals:
        text = source_rows[row["source_index"]]["response"]
        if row.get("error") or hashlib.sha256(text.encode()).hexdigest()!=row["text_sha256"]:
            raise ValueError("Original null data is incomplete or changed")
        texts.append(text)
    args.output.mkdir(mode=0o700)
    public = args.output/"public"
    public.mkdir()
    declaration = {"scope":"Opened-data predictability null development; not fresh confirmation or deployment calibration",
        "script_sha256":sha(Path(__file__)), "candidate_plan_sha256":sha(args.development/"public/plan.json"),
        "power_summary_sha256":sha(args.development/"public/summary.json"),
        "null_plan_sha256":sha(plan_path), "null_results_sha256":sha(args.null_study/"public/results.jsonl"),
        "source_sha256":SOURCE_SHA256, "source":prior["source"], "revision":prior["revision"],
        "attribution":prior["attribution"], "license":prior["license"],
        "planned":500, "unit":"One document across two keys", "family_target":.01,
        "failure_rule":"Retain errors and unavailable selections separately; never replace, retry or fit thresholds",
        "uncertainty":"One-sided 97.5% IID binomial bound is descriptive only; source and topic dependence remain"}
    write(public/"plan.json", declaration)
    rows, fatal = [], None
    started = time.monotonic()
    try:
        from keyprint.backends.mlx import MLXModel
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        backend = MLXModel.load(args.model)
        binding = runtime_binding(max_steps=2048)
        with (public/"results.jsonl").open("x") as stream:
            for original,text in zip(originals,texts,strict=True):
                row = {k:original[k] for k in ("source_index", "text_sha256", "words", "category")}
                begin = time.monotonic()
                try:
                    selection = measure(backend,binding,text)
                    path = args.output/(str(row["source_index"])+".selection.json")
                    write(path, selection)
                    scores = []
                    for key in keys:
                        bits = selected_bits(binding,text,key,selection)
                        if not len(bits):
                            scores.append({"reference_tail":1., "events":0, "unavailable":True})
                            continue
                        value = reference_tail(bits)
                        doubled = reference_tail(bits,double_grid=True)
                        if abs(value["reference_tail"]-doubled["reference_tail"])>1e-10:
                            raise ArithmeticError("Unstable reference-tail grid")
                        scores.append(value)
                    family = two_key_tail([s["reference_tail"] for s in scores])
                    row.update(weighted=scores, family_reference_tail=family, flagged=family<=.01,
                        selected_events=len(selection["selected_positions"]),
                        unavailable=not bool(selection["selected_positions"]), selection_sha256=sha(path),
                        baseline_flagged=original["family_reference_tail"]["linear_10_to_1"]<=.01)
                except Exception as exc:
                    row["error"] = {"type":type(exc).__name__, "message":str(exc)}
                row["seconds"] = time.monotonic()-begin
                rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+"\n")
                stream.flush()
                if len(rows)%25==0:
                    print(json.dumps({"completed":len(rows), "hits":sum(r.get("flagged",False) for r in rows),
                                      "errors":sum("error" in r for r in rows)}),flush=True)
    except Exception as exc:
        fatal = {"type":type(exc).__name__, "message":str(exc)}
    errors = sum("error" in r for r in rows)
    unavailable = sum(r.get("unavailable",False) for r in rows)
    hits = sum(r.get("flagged",False) for r in rows)
    complete = len(rows)==500 and not errors and not unavailable and fatal is None
    summary = {"status":"completed" if complete else "incomplete", "attempts":len(rows), "errors":errors,
               "unavailable":unavailable, "fatal":fatal, "hits":hits,
               "baseline_hits":sum(r.get("baseline_flagged",False) for r in rows),
               "iid_only_upper_97_5":iid_binomial_upper(hits,500) if complete else None,
               "seconds":time.monotonic()-started, "deployment_calibrated":False,
               "scope":declaration["scope"]}
    write(public/"summary.json",summary)
    print(json.dumps(summary),flush=True)
    return 0 if complete else 1


if __name__=="__main__":
    raise SystemExit(main())
