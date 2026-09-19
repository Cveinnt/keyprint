"""Fixed weighted-reference screen across two article-length strata.

No threshold fitting, no model inference, no production detector verdict.
Two fresh keys, unchanged .01 family threshold; one article = one observation.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import secrets
import time

from layer_likelihood import extract
from prepare_wikitext_controls import FILES, LENGTHS, REVISION, sha, write
from validate_null_corpus import iid_binomial_upper
from weighted_null import reference_tail, two_key_tail, WEIGHTS, ALIAS_TARGET, NUMERIC_MARGIN


def load_controls(path):
    manifest = json.loads((path / "public/manifest.json").read_text())
    raw = (path / "texts.jsonl").read_bytes()
    if manifest["revision"] != REVISION or manifest["source_files"] != FILES or sha(raw) != manifest["texts_sha256"]:
        raise ValueError("Pinned source/selection checksum differs")
    rows = [json.loads(line) for line in raw.splitlines()]
    if [{k: v for k, v in r.items() if k != "text"} for r in rows] != manifest["records"]:
        raise ValueError("Source metadata differs")
    for row in rows:
        if sha(row["text"].encode()) != row["text_sha256"] or len(row["text"].split()) != row["words"]:
            raise ValueError("Text checksum or length differs")
    n = manifest["selection"]["planned_documents"]
    if len(rows) != n or len({r["start_row"] for r in rows}) != n or len({r["opening_sha256"] for r in rows}) != n:
        raise ValueError("Missing or repeated articles/openings")
    if Counter(r["words"] for r in rows) != {w: manifest["selection"]["per_length"] for w in LENGTHS}:
        raise ValueError("Length allocation differs")
    return manifest, rows


def panel(rows, expected):
    available = [r for r in rows if "error" not in r]
    hits = sum(r["flagged"] for r in available)
    return {"planned": expected, "available": len(available), "false_hits": hits,
            "unavailable": expected - len(available),
            "iid_only_upper_97_5_percent": iid_binomial_upper(hits, expected)
            if len(available) == expected else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--prior-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, selected = load_controls(args.controls)
    prior = json.loads((args.prior_study / "public/plan.json").read_text())
    keys = [secrets.token_bytes(32) for _ in range(2)]
    commitments = [sha(key) for key in keys]
    if len(set(commitments)) != 2 or set(commitments) & set(prior["key_commitments"]):
        raise ValueError("New independent keys required")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    for i, key in enumerate(keys):
        with os.fdopen(os.open(args.output / f"owner-{i}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(key)
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding = runtime_binding(max_steps=2048)
    plan = {"scope": "New-source article-length screen of the unchanged weighted reference rule; not deployment qualification",
            "controls": manifest, "key_commitments": commitments,
            "binding": binding.profile.identity_receipt(),
            "script_sha256": sha(Path(__file__).read_bytes()),
            "dependencies_sha256": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
                                    ("prepare_wikitext_controls.py", "weighted_null.py", "layer_likelihood.py", "validate_null_corpus.py")},
            "prior_plan_sha256": sha((args.prior_study / "public/plan.json").read_bytes()),
            "weights_integer": list(WEIGHTS), "alias_target": ALIAS_TARGET, "numeric_margin": NUMERIC_MARGIN,
            "primary_rule": "2 * min(two key reference tails) <= .01; unchanged from fresh Dolly screen",
            "target": .01, "unit": "One article across two keys; each article appears in only one length stratum",
            "grid_check": "Double grid for every score; difference must be <= 1e-10",
            "uncertainty": "Each 97.5% one-sided binomial upper bound assumes IID articles. These are descriptive per-panel bounds, not a simultaneous or deployment guarantee. Shared authors/topics, source punctuation artifacts and fixed keys limit generalization.",
            "failure_rule": "Retain every attempt; no retries or row replacement. Missing scores unavailable, never negatives."}
    write(public / "plan.json", plan)
    (public / "ATTRIBUTION.md").write_bytes((args.controls / "public/ATTRIBUTION.md").read_bytes())
    results = []
    started = time.monotonic()
    with (public / "results.jsonl").open("x", encoding="utf-8") as stream:
        for item in selected:
            row = {k: v for k, v in item.items() if k != "text"}
            try:
                scores = []
                for key in keys:
                    bits = extract(binding, item["text"], key)
                    score = reference_tail(bits)
                    second = reference_tail(bits, double_grid=True)
                    delta = abs(score["reference_tail"] - second["reference_tail"])
                    if delta > 1e-10:
                        raise ArithmeticError("Reference tail is not grid stable")
                    scores.append({**score, "double_grid_difference": delta})
                family = two_key_tail([s["reference_tail"] for s in scores])
                row.update(weighted=scores, family_reference_tail=family, flagged=family <= .01)
            except Exception as exc:
                row["error"] = type(exc).__name__
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if len(results) % 250 == 0:
                print(json.dumps({"documents": len(results), "errors": sum("error" in r for r in results)}), flush=True)
    errors = sum("error" in r for r in results)
    summary = {"status": "completed" if not errors else "incomplete", "errors": errors,
               "combined": panel(results, len(selected)),
               "lengths": {str(w): panel([r for r in results if r["words"] == w], manifest["selection"]["per_length"]) for w in LENGTHS},
               "seconds": time.monotonic() - started, "deployment_calibrated": False,
               "scope": plan["scope"], "uncertainty": plan["uncertainty"]}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
