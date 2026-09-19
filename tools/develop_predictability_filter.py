"""One frozen prompt-free entropy filter on all opened weighted-power texts.

No new generation, original prompts, threshold search or SDK changes. Saves
every result and error. Private per-position model diagnostics stay private.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from predictability_filter import measure, selected_bits, PREFIX, MIN_ENTROPY_BITS, CHUNK_SIZE, PAD_TOKEN_ID
from weighted_null import reference_tail, two_key_tail, WEIGHTS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.study / "public/plan.json"
    original_plan = json.loads(plan_path.read_text())
    paths = sorted((args.study / "public").glob("weighted-*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    expected = {(task["source_index"], condition) for task in original_plan["tasks"] for condition in ("ordinary", "marked")}
    if len(rows) != 24 or len(expected) != 24 or {(r["source_index"],r["condition"]) for r in rows} != expected:
        raise ValueError("All twelve original paired cases required")
    for path,row in zip(paths, rows, strict=True):
        if row["id"] != path.stem or row.get("error"):
            raise ValueError("Source identity differs or source is incomplete")
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != original_plan["key_commitments"]:
        raise ValueError("Original key commitments differ")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    declaration = {"scope":"Opened-data development, not fresh confirmation or deployment calibration",
        "script_sha256":sha(Path(__file__)), "dependencies_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("predictability_filter.py", "weighted_null.py")},
        "source_plan_sha256":sha(plan_path), "source_rows_sha256":{p.name:sha(p) for p in paths},
        "fixed_prefix":PREFIX, "entropy_threshold_bits":MIN_ENTROPY_BITS,
        "temperature":.7, "top_k":100, "chunk_size":CHUNK_SIZE,
        "final_batch_padding_token":PAD_TOKEN_ID, "padding_policy":"Fixed 64-position batches; padded future heads discarded, cache discarded after final batch",
        "weights_integer":list(WEIGHTS), "per_key_target":.005, "family_target":.01,
        "selection":"Eligible unseen canonical contexts whose key-blind prefix entropy is >= 0.5 bits",
        "failure_rule":"Retain every row; no generation, retries, threshold search or post-hoc sample replacement",
        "empty_selection":"Unavailable, counted as a miss",
        "next_step_gate":"Only expand to ordinary-corpus controls if at least one extra short marked answer is recovered, overall marked hits do not fall, and no ordinary/wrong-key hits are introduced. This is a development screen, not launch acceptance."}
    write(public / "plan.json", declaration)
    results, fatal = [], None
    started = time.monotonic()
    try:
        from keyprint.backends.mlx import MLXModel
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        backend = MLXModel.load(args.model)
        binding = runtime_binding(max_steps=2048)
        for original in rows:
            row = {k:original[k] for k in ("id", "condition", "key_index", "words", "completion")}
            row["baseline_matching"] = original["matching_flagged"]
            row["baseline_other"] = original["other_flagged"]
            begin = time.monotonic()
            try:
                selection = measure(backend, binding, original["text"])
                write(args.output / (row["id"] + ".selection.json"), selection)
                scores = []
                for key in keys:
                    bits = selected_bits(binding, original["text"], key, selection)
                    if not len(bits):
                        scores.append({"reference_tail":1., "events":0, "unavailable":True})
                        continue
                    value = reference_tail(bits)
                    doubled = reference_tail(bits, double_grid=True)
                    if abs(value["reference_tail"] - doubled["reference_tail"]) > 1e-10:
                        raise ArithmeticError("Unstable reference-tail grid")
                    scores.append(value)
                row.update(selected_events=len(selection["selected_positions"]),
                    all_eligible_events=len(selection["eligible_positions"]), weighted=scores,
                    family_reference_tail=two_key_tail([s["reference_tail"] for s in scores]),
                    matching_flagged=scores[row["key_index"]]["reference_tail"] <= .005,
                    other_flagged=scores[1-row["key_index"]]["reference_tail"] <= .005,
                    selection_sha256=sha(args.output/(row["id"]+".selection.json")))
            except Exception as exc:
                row["error"] = {"type":type(exc).__name__, "message":str(exc)}
            row["seconds"] = time.monotonic() - begin
            results.append(row)
            write(public/(row["id"]+".json"), row)
            print(json.dumps({k:v for k,v in row.items() if k != "weighted"}), flush=True)
    except Exception as exc:
        fatal = {"type":type(exc).__name__, "message":str(exc)}
    errors = sum("error" in r for r in results)
    groups = {condition:{field:sum(r.get(field,False) for r in results if r["condition"]==condition)
              for field in ("matching_flagged", "other_flagged", "baseline_matching", "baseline_other")}
              for condition in ("ordinary", "marked")}
    short = [r for r in results if r["condition"]=="marked" and 100 <= r["words"] <= 400]
    short_hits = sum(r.get("matching_flagged",False) for r in short)
    short_baseline = sum(r["baseline_matching"] for r in short)
    complete = len(results)==24 and not errors and fatal is None
    gate = bool(complete and short_hits>short_baseline and groups["marked"]["matching_flagged"]>=groups["marked"]["baseline_matching"]
                and not groups["marked"]["other_flagged"] and not groups["ordinary"]["matching_flagged"] and not groups["ordinary"]["other_flagged"])
    summary = {"status":"completed" if complete else "incomplete", "attempts":len(results), "errors":errors, "fatal":fatal,
               "groups":groups, "short_marked":{"count":len(short), "hits":short_hits, "baseline_hits":short_baseline},
               "expand_to_null_controls":gate, "seconds":time.monotonic()-started,
               "deployment_calibrated":False, "scope":declaration["scope"]}
    write(public/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
