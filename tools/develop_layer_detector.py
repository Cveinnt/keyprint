"""Fit a conditional-layer research score and diagnose already-opened controls.

This is development, not fresh confirmation. No model generation or threshold
search. Keep all attempts and failures. Untouched source records remain unused.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from compare_score_baselines import score
from layer_likelihood import extract, fit, log_evidence, FRACTIONS, RIDGE, REFERENCE
from validate_null_corpus import SOURCE_SHA256, iid_binomial_upper, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, nargs=2, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--null-study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.source) != SOURCE_SHA256:
        raise ValueError("Pinned corpus checksum differs")
    inputs, training = {}, []
    for root in args.training:
        key = (root / "owner.key").read_bytes()
        paths = sorted((root / "public").glob("*-marked.json"))
        if len(paths) != 12:
            raise ValueError("Each training study must contain twelve marked responses")
        for path in paths:
            row = json.loads(path.read_text())
            if row.get("error"):
                raise ValueError("Training study has unresolved failures")
            training.append((row, key))
            inputs[str(path)] = sha(path)
        inputs[str(root / "owner.key")] = sha(root / "owner.key")
        inputs[str(root / "public/plan.json")] = sha(root / "public/plan.json")
    power_paths = sorted((args.power / "public").glob("power-*.json"))
    if len(power_paths) != 24:
        raise ValueError("Complete source power study required")
    power_keys = [(args.power / f"owner-{i}.key").read_bytes() for i in range(2)]
    null_keys = [(args.null_study / f"heldout-{i}.key").read_bytes() for i in range(2)]
    if any(k == t for k in power_keys + null_keys for _, t in training):
        raise ValueError("Development evaluation keys must differ from training keys")
    null_rows = [json.loads(line) for line in (args.null_study / "public/heldout.jsonl").read_text().splitlines()]
    if len(null_rows) != 500 or any(r.get("error") for r in null_rows):
        raise ValueError("Complete 500-response opened null study required")
    for path in [*power_paths, args.null_study / "public/heldout.jsonl"]:
        inputs[str(path)] = sha(path)
    args.output.mkdir(mode=0o700); public = args.output / "public"; public.mkdir()
    cutoff = math.log(2 / .01)
    plan = {"scope": "Opened-data detector development; not fresh confirmation or deployment qualification",
            "script_sha256": sha(Path(__file__)), "detector_sha256": sha(Path(__file__).with_name("layer_likelihood.py")),
            "inputs_sha256": inputs, "source_sha256": SOURCE_SHA256,
            "reference": REFERENCE, "training_responses": 24, "ridge": RIDGE,
            "fractions": list(FRACTIONS), "fixed_log_evidence_cutoff": cutoff,
            "cutoff_rule": "log(2 / .01), inclusive; union bound over two keys under independent fair ideal-PRF null bits",
            "limitations": "Not a posterior, authorship claim or fixed-key deployment guarantee; training and evaluation texts already opened",
            "no_tuning": "No cutoff search, refitting after evaluation, resampling, or excluding failures"}
    write(public / "plan.json", plan)
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding = runtime_binding(max_steps=2048)
    started = time.monotonic()
    matrices = [extract(binding, row["text"], key) for row, key in training]
    model = fit(np.concatenate(matrices))
    model["binding"] = binding.profile.identity_receipt()
    write(public / "model.json", model)
    print(json.dumps({"fitted_events": model["fit_events"], "fit_seconds": time.monotonic() - started}), flush=True)
    rows = []
    for path in power_paths:
        source = json.loads(path.read_text())
        row = {k: source[k] for k in ("id", "condition", "key_index", "completion", "words")}
        try:
            values = []
            for i, key in enumerate(power_keys):
                bits = extract(binding, source["text"], key)
                label = "matching" if i == source["key_index"] else "other"
                if not np.isclose(score(bits, np.ones(30)), source["scores"][label]["uniform"], rtol=1e-12, atol=1e-12):
                    raise ValueError("Original event score did not reproduce")
                values.append(log_evidence(bits, model))
            row.update(log_evidence=values, matching_above=values[source["key_index"]] >= cutoff,
                       other_above=values[1 - source["key_index"]] >= cutoff, any_key_above=max(values) >= cutoff)
        except Exception as exc:
            row["error"] = type(exc).__name__
        rows.append(row)
    write(public / "power.json", rows)
    corpus = [json.loads(line) for line in args.source.read_bytes().splitlines()]
    controls = []
    with (public / "null.jsonl").open("x") as stream:
        for original in null_rows:
            row = {k: original[k] for k in ("source_index", "text_sha256", "category", "words")}
            try:
                text = corpus[row["source_index"]]["response"]
                if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
                    raise ValueError("Source response differs")
                values = []
                for i, key in enumerate(null_keys):
                    bits = extract(binding, text, key)
                    if not np.isclose(score(bits, np.ones(30)), original["scores"][i]["uniform"], rtol=1e-12, atol=1e-12):
                        raise ValueError("Original null event score did not reproduce")
                    values.append(log_evidence(bits, model))
                row.update(log_evidence=values, any_key_above=max(values) >= cutoff)
            except Exception as exc:
                row["error"] = type(exc).__name__
            controls.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
            if len(controls) % 50 == 0:
                print(json.dumps({"controls": len(controls), "errors": sum('error' in r for r in controls)}), flush=True)
    errors = sum('error' in r for r in rows + controls)
    hits = sum(r.get("any_key_above", False) for r in controls)
    summary = {"status": "completed" if not errors else "incomplete", "errors": errors,
               "null": {"hits": hits, "planned": 500,
                        "iid_only_upper_97_5_percent": iid_binomial_upper(hits, 500) if not errors else None},
               "power": {condition: {"planned": 12, "matching": sum(r.get("matching_above", False) for r in rows if r["condition"] == condition),
                                      "other": sum(r.get("other_above", False) for r in rows if r["condition"] == condition)} for condition in ("ordinary", "marked")},
               "seconds": time.monotonic() - started, "deployment_calibrated": False, "scope": plan["scope"]}
    write(public / "summary.json", summary); print(json.dumps(summary), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
