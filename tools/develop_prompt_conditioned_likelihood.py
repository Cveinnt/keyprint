"""Test original-prompt likelihood replay on all 24 opened paired responses.

Development only. Uses visible text, original user prompts, pinned model and
owner keys; does not read private generation journals or saved model heads.
No score tuning, sample exclusions, fresh confirmation or SDK promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from develop_surrogate_likelihood import sha, write
from prompt_conditioned_likelihood import measure
from surrogate_likelihood import score, CHUNK_SIZE, MIXTURE, LOG_CUTOFF


def sources(study):
    plan_path = study / "public/plan.json"
    plan = json.loads(plan_path.read_text())
    paths = sorted((study / "public").glob("weighted-*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    tasks = {t["source_index"]: t for t in plan["tasks"]}
    expected = {(i, c) for i in tasks for c in ("ordinary", "marked")}
    if len(tasks) != 12 or len(rows) != 24 or {(r["source_index"], r["condition"]) for r in rows} != expected:
        raise ValueError("Every source pair required exactly once")
    if any(r.get("error") or r["id"] != p.stem for r, p in zip(rows, paths, strict=True)):
        raise ValueError("Source rows incomplete or misidentified")
    return plan_path, plan, paths, rows, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan_path, source_plan, paths, rows, tasks = sources(args.study)
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != source_plan["key_commitments"]:
        raise ValueError("Key commitments differ")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    write(public / "plan.json", {"scope": __doc__, "script_sha256": sha(Path(__file__)),
        "dependencies_sha256": {n: sha(Path(__file__).with_name(n)) for n in (
            "prompt_conditioned_likelihood.py", "surrogate_likelihood.py", "predictability_filter.py", "log_tournament.py")},
        "source_plan_sha256": sha(plan_path), "source_rows_sha256": {p.name: sha(p) for p in paths},
        "conditioning": "Exact original task prompt; user-only non-thinking chat template",
        "chunk_size": CHUNK_SIZE, "temperature": .7, "top_k": 100,
        "mixture": MIXTURE, "fixed_log_cutoff": LOG_CUTOFF, "comparison": "inclusive",
        "gate": "At least 10/12 marked, 4/5 short marked; zero ordinary matching, ordinary other and marked other hits; all 24 retained",
        "statistical_scope": "Opened development cases only; no calibrated false-positive bound",
        "private_generation_data_used": False, "new_text_generation": False,
        "finite_precision": "Fixed batch differs from single-token generation; exact numerical parity is not assumed"})
    started = time.monotonic()
    results, fatal = [], None
    try:
        from keyprint.backends.mlx import MLXModel, ASSETS
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        from keyprint import sampling
        binding = runtime_binding(max_steps=2048)
        backend = MLXModel.load(args.model)
        write(public / "identity.json", {"model_assets": ASSETS,
            "profile": binding.profile.identity_receipt(), "sampling": sampling.identity()})
        for original in rows:
            row = {k: original[k] for k in ("id", "condition", "key_index", "words", "completion")}
            begin = time.monotonic()
            try:
                measurement = measure(backend, binding, original["text"], tasks[original["source_index"]]["prompt"])
                head_path = args.output / (row["id"] + ".heads.json")
                write(head_path, measurement)
                values = [score(binding.profile, key, measurement) for key in keys]
                score_path = args.output / (row["id"] + ".scores.json")
                write(score_path, values)
                row.update(working_log_ratios=[v["working_log_ratio"] for v in values],
                    matching_flagged=values[row["key_index"]]["flagged"], other_flagged=values[1-row["key_index"]]["flagged"],
                    scored_events=values[0]["scored_events"], outside_support=values[0]["outside_support"],
                    heads_sha256=sha(head_path), scores_sha256=sha(score_path))
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
    complete = len(results) == 24 and not fatal and not any("error" in r for r in results)
    gate = (complete and groups["marked"]["matching_flagged"] >= 10 and len(short) == 5
            and sum(r.get("matching_flagged", False) for r in short) >= 4
            and not any(groups[c][field] for c, field in (
                ("ordinary", "matching_flagged"), ("ordinary", "other_flagged"), ("marked", "other_flagged"))))
    summary = {"status": "completed" if complete else "incomplete", "attempts": len(results),
        "errors": sum("error" in r for r in results), "fatal": fatal, "groups": groups,
        "short_marked": {"count": len(short), "hits": sum(r.get("matching_flagged", False) for r in short)},
        "power_gate_passed": bool(gate), "null_expansion_requires_audit": True,
        "seconds": time.monotonic() - started, "deployment_calibrated": False,
        "prompt_free": False, "private_generation_data_used": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
