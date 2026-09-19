"""One frozen recipe on opened data; no fitting to evaluation scores.

Retains 48 real-model outputs and 1,000 ordinary controls from completed studies.
Primary family threshold .005; .01 is descriptive only, never choose-either.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from layer_likelihood import extract
from learned_layer_weights import fit_weights, reference_tail, MAX_WEIGHT, PRIOR_COUNT
from validate_null_corpus import SOURCE_SHA256, write
from weighted_null import reference_tail as original_tail, two_key_tail


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, field, expected):
    available = [r for r in rows if "error" not in r]
    hits = sum(r[field] for r in available)
    return {"planned": expected, "available": len(available), "hits": hits,
            "unavailable": expected - len(available)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, nargs=2, required=True)
    parser.add_argument("--power", type=Path, nargs=2, required=True)
    parser.add_argument("--null-old", type=Path, required=True)
    parser.add_argument("--null-fresh", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.source) != SOURCE_SHA256:
        raise ValueError("Pinned source differs")
    inputs, training, power, controls = {}, [], [], []

    def record(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())

    training_keys = []
    for root in args.training:
        key = (root / "owner.key").read_bytes()
        training_keys.append(key)
        inputs[str(root / "owner.key")] = sha(root / "owner.key")
        paths = sorted((root / "public").glob("*-marked.json"))
        if len(paths) != 12:
            raise ValueError("Twelve marked training outputs per study required")
        for path in paths:
            row = record(path)
            if row.get("error"):
                raise ValueError("Unresolved training errors")
            training.append((row, key))
    for index, root in enumerate(args.power):
        keys = [(root / f"owner-{i}.key").read_bytes() for i in range(2)]
        if len(set(keys)) != 2 or any(k in training_keys for k in keys):
            raise ValueError("Evaluation keys must be distinct from training keys")
        for i in range(2):
            inputs[str(root / f"owner-{i}.key")] = sha(root / f"owner-{i}.key")
        paths = sorted((root / "public").glob("power-*.json" if index == 0 else "weighted-*.json"))
        if len(paths) != 24:
            raise ValueError("Each power study must retain 24 outputs")
        for path in paths:
            power.append((index, record(path), keys))
    corpus = [json.loads(line) for line in args.source.read_bytes().splitlines()]
    for index, root in enumerate((args.null_old, args.null_fresh)):
        filename = "heldout.jsonl" if index == 0 else "results.jsonl"
        source = root / "public" / filename
        inputs[str(source)] = sha(source)
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        keys = [(root / f"{'heldout' if index == 0 else 'owner'}-{i}.key").read_bytes() for i in range(2)]
        if len(rows) != 500 or len(set(keys)) != 2 or any(k in training_keys for k in keys):
            raise ValueError("Complete opened null study with nontraining keys required")
        for i in range(2):
            path = root / f"{'heldout' if index == 0 else 'owner'}-{i}.key"
            inputs[str(path)] = sha(path)
        for row in rows:
            text = corpus[row["source_index"]]["response"]
            if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"] or row.get("error"):
                raise ValueError("Original null data differs or is incomplete")
            controls.append((index, row, text, keys))
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    plan = {"scope": "Opened-data weight development; not fresh confirmation or deployment qualification",
            "script_sha256": sha(Path(__file__)),
            "dependencies_sha256": {n: sha(Path(__file__).with_name(n)) for n in
                                    ("learned_layer_weights.py", "layer_likelihood.py", "weighted_null.py", "validate_null_corpus.py", "compare_score_baselines.py")},
            "inputs_sha256": inputs, "source_sha256": SOURCE_SHA256,
            "training_responses": 24, "training_recipe": "Pool training events; smoothed marginal positive log odds, clipped to [.5,.75], integer normalized maximum 256 and minimum 1",
            "max_weight": MAX_WEIGHT, "prior_count": PRIOR_COUNT,
            "primary_family_target": .005, "secondary_descriptive_target": .01,
            "rationale": "Operating margin below a future 1% empirical ceiling; development only. A nominal 1% test should not be expected to have its confidence upper bound below 1%.",
            "comparison": "Unchanged linear weights at the same two operating points; never choose either scorer or threshold as one test",
            "failure_rule": "Retain all rows and errors; no replacement, refitting, threshold search or new inference",
            "grid_check": "Double every learned-weight FFT grid; require difference <=1e-10",
            "training_dependence": "Events may overlap across training documents. Fitted weights are development coefficients, not independent-trial parameter confidence estimates."}
    write(public / "plan.json", plan)
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding = runtime_binding(max_steps=2048)
    started = time.monotonic()
    model = fit_weights(np.concatenate([extract(binding, row["text"], key) for row, key in training]))
    model["binding"] = binding.profile.identity_receipt()
    write(public / "model.json", model)
    weights = model["weights_integer"]

    def evaluate(text, keys):
        learned, original = [], []
        for key in keys:
            bits = extract(binding, text, key)
            value = reference_tail(bits, weights)
            doubled = reference_tail(bits, weights, double_grid=True)
            delta = abs(value["reference_tail"] - doubled["reference_tail"])
            if delta > 1e-10:
                raise ArithmeticError("Learned-weight tail is not grid stable")
            learned.append({**value, "double_grid_difference": delta})
            original.append(original_tail(bits))
        family = two_key_tail([s["reference_tail"] for s in learned])
        baseline = two_key_tail([s["reference_tail"] for s in original])
        return {"learned": learned, "original": original, "family_reference_tail": family,
                "original_family_reference_tail": baseline, "primary_hit": family <= .005,
                "descriptive_hit": family <= .01, "original_primary_hit": baseline <= .005,
                "original_descriptive_hit": baseline <= .01}

    results = []
    for study, original, keys in power:
        row = {k: original[k] for k in ("id", "condition", "key_index", "completion", "words")}
        row["study"] = study
        try:
            if original.get("error"):
                raise ValueError("Original inference failed")
            row.update(evaluate(original["text"], keys))
            if study == 1 and row["original_family_reference_tail"] != original["family_reference_tail"]:
                raise ValueError("Original reference tail differs")
            index = original["key_index"]
            row["matching_hit"] = row["learned"][index]["reference_tail"] <= .0025
            row["other_hit"] = row["learned"][1 - index]["reference_tail"] <= .0025
        except Exception as exc:
            row["error"] = type(exc).__name__
        results.append(row)
    write(public / "power.json", results)
    null = []
    with (public / "null.jsonl").open("x") as stream:
        for study, original, text, keys in controls:
            row = {k: original[k] for k in ("source_index", "text_sha256", "category", "words")}
            row["study"] = study
            try:
                row.update(evaluate(text, keys))
                if study == 1 and row["original_family_reference_tail"] != original["family_reference_tail"]["linear_10_to_1"]:
                    raise ValueError("Original null tail differs")
            except Exception as exc:
                row["error"] = type(exc).__name__
            null.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if len(null) % 100 == 0:
                print(json.dumps({"controls": len(null), "errors": sum("error" in r for r in null)}), flush=True)
    errors = sum("error" in r for r in results + null)
    fields = ("primary_hit", "descriptive_hit", "original_primary_hit", "original_descriptive_hit")
    summary = {"status": "completed" if not errors else "incomplete", "errors": errors,
               "null": {f: summarize(null, f, 1000) for f in fields},
               "power": {str(i): {c: {f: summarize([r for r in results if r["study"] == i and r["condition"] == c], f, 12) for f in fields} for c in ("ordinary", "marked")} for i in (0, 1)},
               "seconds": time.monotonic() - started, "scope": plan["scope"],
               "deployment_calibrated": False, "fresh_confirmation": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
