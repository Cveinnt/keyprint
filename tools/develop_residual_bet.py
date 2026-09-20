"""One frozen model-centered score on all 24 opened paired responses.

Reuses retained key-blind predictive heads; no new text, random draws, model
inference, coefficient fitting or threshold search. Development evidence only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from residual_bet import score, STAKES, LOG_CUTOFF
from predictability_filter import PREFIX, CHUNK_SIZE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "heads", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source_plan = args.study / "public/plan.json"
    original = json.loads(source_plan.read_text())
    paths = sorted((args.study / "public").glob("weighted-*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    expected = {(t["source_index"], c) for t in original["tasks"] for c in ("ordinary", "marked")}
    if (len(rows) != 24 or len(expected) != 24
            or {(r["source_index"], r["condition"]) for r in rows} != expected
            or any(r.get("error") or r["id"] != p.stem for r, p in zip(rows, paths, strict=True))):
        raise ValueError("All 24 exact completed source cases are required")
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != original["key_commitments"]:
        raise ValueError("Source keys differ")
    old_plan = json.loads((args.heads / "public/plan.json").read_text())
    old_summary = json.loads((args.heads / "public/summary.json").read_text())
    old_identity = json.loads((args.heads / "public/identity.json").read_text())
    from keyprint.backends.mlx import ASSETS
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from keyprint._engine.legacy._impl.research import token_source_sparse_execution
    from keyprint import sampling
    binding = runtime_binding(max_steps=2048)
    if (old_plan["source_plan_sha256"] != sha(source_plan)
            or old_plan["source_rows_sha256"] != {p.name: sha(p) for p in paths}
            or old_plan["prefix"] != PREFIX or old_plan["chunk_size"] != CHUNK_SIZE
            or old_plan["temperature"] != .7 or old_plan["top_k"] != 100
            or old_identity["model_assets"] != ASSETS
            or old_identity["profile"] != binding.profile.identity_receipt()
            or old_identity["sampling"] != sampling.identity()
            or old_summary["status"] != "completed" or old_summary["attempts"] != 24):
        raise ValueError("Complete archived heads with identical predictive binding required")
    measurements = []
    head_paths = []
    for row in rows:
        path = args.heads / (row["id"] + ".heads.json")
        archived = json.loads((args.heads / "public" / (row["id"] + ".json")).read_text())
        measurement = json.loads(path.read_text())
        if (sha(path) != archived["heads_sha256"]
                or measurement["token_ids"] != list(binding.encode_visible(row["text"]))
                or measurement["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest()
                or measurement["original_prompt_used"] is not False
                or measurement["key_used_for_model_inference"] is not False):
            raise ValueError("Retained heads or causal text identity differ")
        head_paths.append(path)
        measurements.append(measurement)
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    plan = {"scope": __doc__, "script_sha256": sha(Path(__file__)),
        "dependencies_sha256": {n: sha(Path(__file__).with_name(n)) for n in
            ("residual_bet.py", "surrogate_likelihood.py", "predictability_filter.py", "log_tournament.py")},
        "bit_source_sha256": sha(Path(token_source_sparse_execution.__file__)),
        "source_plan_sha256": sha(source_plan), "source_rows_sha256": {p.name: sha(p) for p in paths},
        "archived_head_plan_sha256": sha(args.heads / "public/plan.json"),
        "archived_head_identity_sha256": sha(args.heads / "public/identity.json"),
        "heads_sha256": {p.name: sha(p) for p in head_paths},
        "stakes": STAKES, "stake_weight": "uniform whole-document mixture, not a maximum",
        "fixed_log_cutoff": LOG_CUTOFF, "comparison": "inclusive",
        "residual": "observed canonical-label bit minus predictive nonempty-label weighted bit mean",
        "factor": "1 + stake*residual per fresh context and layer; multiply within each stake, then average stakes",
        "skip": "excluded labels, repeated canonical contexts and out-of-support observed tokens contribute factor one; literal context still advances",
        "null_interpretation": "Ideal independent random-key rationale only; fixed HMAC key deployment, key-dependent text and floating-point error rates unqualified",
        "gate": "At least 10/12 marked, at least 4/5 short marked, zero ordinary or wrong-key flags, complete audit before null expansion",
        "references": ["https://arxiv.org/abs/2602.14286", "https://proceedings.mlr.press/v258/li25d.html"],
        "reference_scope": "Background on model-aware and betting-process detection, not this implementation or a novelty claim"}
    write(public / "plan.json", plan)
    write(public / "identity.json", old_identity)
    results = []
    started = time.monotonic()
    for original, measurement, path in zip(rows, measurements, head_paths, strict=True):
        row = {k: original[k] for k in ("id", "condition", "key_index", "words", "completion")}
        begin = time.monotonic()
        try:
            values = [score(binding.profile, k, measurement) for k in keys]
            detail = args.output / (row["id"] + ".scores.json")
            write(detail, values)
            row.update(working_log_evidence=[v["working_log_evidence"] for v in values],
                matching_flagged=values[row["key_index"]]["flagged"], other_flagged=values[1-row["key_index"]]["flagged"],
                scored_events=values[0]["scored_events"], outside_support=values[0]["outside_support"],
                heads_sha256=sha(path), scores_sha256=sha(detail))
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        row["seconds"] = time.monotonic() - begin
        results.append(row)
        write(public / (row["id"] + ".json"), row)
        print(json.dumps(row), flush=True)
    groups = {c: {field: sum(r.get(field, False) for r in results if r["condition"] == c)
        for field in ("matching_flagged", "other_flagged")} for c in ("ordinary", "marked")}
    short = [r for r in results if r["condition"] == "marked" and 100 <= r["words"] <= 400]
    short_hits = sum(r.get("matching_flagged", False) for r in short)
    errors = sum("error" in r for r in results)
    gate = (not errors and groups["marked"]["matching_flagged"] >= 10 and short_hits >= 4
        and not groups["ordinary"]["matching_flagged"] and not groups["ordinary"]["other_flagged"]
        and not groups["marked"]["other_flagged"])
    summary = {"status": "incomplete" if errors else "completed", "attempts": len(results), "errors": errors,
        "groups": groups, "short_marked": {"count": len(short), "hits": short_hits},
        "power_gate_passed": bool(gate), "null_expansion_requires_audit": True,
        "seconds": time.monotonic() - started, "deployment_calibrated": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
