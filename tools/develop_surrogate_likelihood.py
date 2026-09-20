"""Freeze and test one working score on all opened paired Qwen responses.

This is development evidence, not an independent power or false-positive study.
No new text generation, cutoff search, per-case tuning or omitted failed cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from surrogate_likelihood import measure, score, PREFIX, CHUNK_SIZE, MIXTURE, LOG_CUTOFF


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-heads", type=Path, help="Retain and rescore a complete earlier head collection; no new inference")
    args = parser.parse_args()
    plan_path = args.study / "public/plan.json"
    original_plan = json.loads(plan_path.read_text())
    paths = sorted((args.study / "public").glob("weighted-*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    expected = {(task["source_index"], c) for task in original_plan["tasks"] for c in ("ordinary", "marked")}
    if len(rows) != 24 or len(expected) != 24 or {(r["source_index"], r["condition"]) for r in rows} != expected:
        raise ValueError("Every original ordinary/marked case required")
    if any(r.get("error") or r["id"] != p.stem for r, p in zip(rows, paths, strict=True)):
        raise ValueError("Source study incomplete or identities differ")
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != original_plan["key_commitments"]:
        raise ValueError("Source key commitments differ")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    declaration = {"scope": __doc__, "script_sha256": sha(Path(__file__)),
        "dependencies_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                                ("surrogate_likelihood.py", "predictability_filter.py", "log_tournament.py")},
        "source_plan_sha256": sha(plan_path), "source_rows_sha256": {p.name: sha(p) for p in paths},
        "prefix": PREFIX, "chunk_size": CHUNK_SIZE, "temperature": .7, "top_k": 100,
        "mixture": MIXTURE, "fixed_log_cutoff": LOG_CUTOFF, "comparison": "inclusive",
        "score": "sum log(0.5 + 0.5*q_key(token)/p_surrogate(token)) over first eligible contexts",
        "zero_support": "factor 1; still advance the literal context",
        "numerics": "log-space ideal tournament; no probability flooring; finite-precision statistical validity remains unqualified",
        "null_interpretation": "Only an ideal independent-PRF normalization rationale. No fixed-key, calibrated, posterior or anytime guarantee.",
        "gate": "Require >=10/12 marked and >3/5 short marked hits, zero ordinary/wrong-key hits, complete audit before expanding to null controls",
        "reference": "https://arxiv.org/html/2609.15657v1; likelihood-ratio family background, not an implementation or novelty claim"}
    write(public / "plan.json", declaration)
    started = time.monotonic()
    results, fatal = [], None
    try:
        from keyprint.backends.mlx import MLXModel, ASSETS
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        from keyprint._engine.legacy._impl.research import token_source_sparse_execution
        from keyprint import sampling
        binding = runtime_binding(max_steps=2048)
        backend = None
        old_plan = None
        if args.reuse_heads:
            old_plan = json.loads((args.reuse_heads / "public/plan.json").read_text())
            old_identity = json.loads((args.reuse_heads / "public/identity.json").read_text())
            if (old_plan["source_plan_sha256"] != declaration["source_plan_sha256"]
                    or old_plan["source_rows_sha256"] != declaration["source_rows_sha256"]
                    or old_plan["prefix"] != PREFIX or old_plan["chunk_size"] != CHUNK_SIZE
                    or old_plan["temperature"] != .7 or old_plan["top_k"] != 100
                    or old_identity["model_assets"] != ASSETS
                    or old_identity["profile"] != binding.profile.identity_receipt()
                    or old_identity["sampling"] != sampling.identity()):
                raise ValueError("Archived model-head identity differs")
            old_files = [args.reuse_heads / (r["id"] + ".heads.json") for r in rows]
            write(public / "reused-heads.json", {"source_plan_sha256": sha(args.reuse_heads / "public/plan.json"),
                "source_summary_sha256": sha(args.reuse_heads / "public/summary.json"),
                "heads_sha256": {p.name: sha(p) for p in old_files}, "new_model_inference": False})
        else:
            backend = MLXModel.load(args.model)
        write(public / "identity.json", {"model_assets": ASSETS, "profile": binding.profile.identity_receipt(),
            "sampling": sampling.identity(), "transform_source_sha256": sha(Path(token_source_sparse_execution.__file__))})
        for original in rows:
            row = {k: original[k] for k in ("id", "condition", "key_index", "words", "completion")}
            begin = time.monotonic()
            try:
                if args.reuse_heads:
                    measurement = json.loads((args.reuse_heads / (row["id"] + ".heads.json")).read_text())
                    if (measurement["token_ids"] != list(binding.encode_visible(original["text"]))
                            or measurement["text_sha256"] != hashlib.sha256(original["text"].encode()).hexdigest()):
                        raise ValueError("Archived heads belong to different text")
                else:
                    measurement = measure(backend, binding, original["text"])
                path = args.output / (row["id"] + ".heads.json")
                write(path, measurement)
                values = [score(binding.profile, k, measurement) for k in keys]
                detail = args.output / (row["id"] + ".scores.json")
                write(detail, values)
                row.update(working_log_ratios=[v["working_log_ratio"] for v in values],
                    matching_flagged=values[row["key_index"]]["flagged"],
                    other_flagged=values[1-row["key_index"]]["flagged"],
                    scored_events=values[0]["scored_events"], outside_support=values[0]["outside_support"],
                    heads_sha256=sha(path), scores_sha256=sha(detail))
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            row["seconds"] = time.monotonic() - begin
            results.append(row)
            write(public / (row["id"] + ".json"), row)
            print(json.dumps(row), flush=True)
    except Exception as exc:
        fatal = {"type": type(exc).__name__, "message": str(exc)}
    groups = {c: {field: sum(r.get(field, False) for r in results if r["condition"] == c)
                  for field in ("matching_flagged", "other_flagged")} for c in ("ordinary", "marked")}
    short = [r for r in results if r["condition"] == "marked" and 100 <= r["words"] <= 400]
    short_hits = sum(r.get("matching_flagged", False) for r in short)
    complete = len(results) == 24 and not fatal and not any("error" in r for r in results)
    gate = (complete and groups["marked"]["matching_flagged"] >= 10 and short_hits > 3
            and not groups["ordinary"]["matching_flagged"] and not groups["ordinary"]["other_flagged"]
            and not groups["marked"]["other_flagged"])
    summary = {"status": "completed" if complete else "incomplete", "attempts": len(results),
        "errors": sum("error" in r for r in results), "fatal": fatal, "groups": groups,
        "short_marked": {"count": len(short), "hits": short_hits},
        "power_gate_passed": bool(gate), "null_expansion_requires_audit": True,
        "seconds": time.monotonic() - started, "deployment_calibrated": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
