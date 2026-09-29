"""Continue the interrupted source-grounded replay under a separate resource plan.

No new randomness, output replacement or SDK changes. Measurements compare the
base and marked distributions at the same recorded prefix, not counterfactual
future text or a causal estimate of semantic harm.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from audit_pydantic_ai import audit_journal
from multikey_prose_pilot import digest, save
from replay_wide_mlx import ordinary_weights, replay_draw


from diagnose_source_sampling import distribution_metrics, aggregate
from summarize_source_sampling import summarize


def validate_prefix(rows, progress, trace_reader):
    if not 0 < len(progress) < len(rows) or len(rows) != 128:
        raise ValueError("Require a nonempty incomplete prefix of the 128-row study")
    fields = ("review_id", "case", "key_slot", "condition")
    for row, result in zip(rows, progress):
        trace = trace_reader(row["review_id"])
        if (any(row[k] != result[k] for k in fields) or result["passed"] is not True
                or result["matched_steps"] != row["tokens"]
                or result["model_calls"] != row["model_calls"]
                or len(trace) != row["tokens"] or not trace
                or any(type(s["selected_is_eos"]) is not bool for s in trace)
                or not trace[-1]["selected_is_eos"]
                or any(s["selected_is_eos"] for s in trace[:-1])
                or any(not math.isfinite(v) for s in trace for v in s.values())
                or aggregate(trace) != result["diagnostics"]):
            raise ValueError("Original completed prefix is missing, changed, failed or reordered")
    return rows[len(progress):]


def validate_preflight(preflight, supervisor, study_plan_hash, runs_hash, first_ids):
    expected_scripts = {'preflight_guarded_source.py', 'replay_wide_mlx.py',
        'audit_pydantic_ai.py', 'multikey_prose_pilot.py', 'memory_watchdog.py', 'mlx_guarded_worker.py'}
    if (set(preflight['plan']['scripts_sha256']) != expected_scripts
            or preflight['plan']['cache_limit_bytes'] != 0
            or preflight['plan']['rows'] != first_ids
            or preflight["passed"] is not True or preflight["matched_steps"] != 16
            or [r["review_id"] for r in preflight["rows"]] != first_ids
            or any(r["passed"] is not True or r["matched_steps"] != 8 for r in preflight["rows"])
            or preflight["plan"]["plan_sha256"] != study_plan_hash
            or preflight["plan"]["runs_sha256"] != runs_hash
            or supervisor["status"] != "completed" or supervisor["exit_code"] != 0
            or supervisor["cleanup_verified"] is not True
            or not 0 < supervisor["limit_bytes"] <= 10*1024**3
            or not 0 < supervisor["peak_footprint_bytes"] <= supervisor["limit_bytes"]):
        raise ValueError("Require the successful bounded first-pair preflight for this study")
    for name, expected in preflight["plan"]["scripts_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Preflight implementation changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    p.add_argument("--incident", type=Path, required=True)
    args = p.parse_args()
    if (str(os.getppid()) != os.environ.get("KEYPRINT_WATCHDOG_PID")
            or "KEYPRINT_MLX_MEMORY_LOG" not in os.environ):
        raise RuntimeError("External watchdog and guarded MLX wrapper required")
    import mlx.core as mx
    if mx.set_cache_limit(0) != 0:
        raise RuntimeError("MLX cache must already be disabled")
    public, private = args.root / "public", args.root / "private"
    read = lambda path: json.loads(path.read_text())
    plan, identity, rows = read(public / "plan.json"), read(public / "identity.json"), read(private / "runs.json")
    audit = read(public / "receipt-audit.json")
    signature = lambda r: (r["key_slot"], r["case"], r["condition"])
    if (len(rows) != 128 or [signature(r) for r in rows] != [signature(r) for r in plan["schedule"]]
            or audit["verified"] != 128 or audit["failures"]
            or audit["runs_sha256"] != digest(private / "runs.json")
            or audit["plan_sha256"] != digest(public / "plan.json")):
        raise ValueError("Require complete audited source-grounded study")
    original_folder = private / "sampling-diagnosis"
    original_plan = read(public / "sampling-diagnosis-plan.json")
    progress_file = original_folder / "progress.json"
    previous = read(progress_file)
    trace_reader = lambda rid: read(original_folder / (rid + ".json"))
    remaining = validate_prefix(rows, previous, trace_reader)
    if (original_plan["runs_sha256"] != digest(private / "runs.json")
            or original_plan["study_plan_sha256"] != digest(public / "plan.json")
            or original_plan["script_sha256"] != digest(Path(__file__).with_name("diagnose_source_sampling.py"))):
        raise ValueError("Original diagnostic commitment differs")
    for name, expected in original_plan["helper_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Original diagnostic helper differs")
    validate_preflight(read(args.preflight), read(args.supervisor), digest(public / "plan.json"),
        digest(private / "runs.json"), [r["review_id"] for r in rows[:2]])
    read(args.incident)  # Require a retained, readable incident receipt.
    output = args.output
    output.mkdir(mode=0o700)
    helpers = ("diagnose_source_sampling.py", "replay_wide_mlx.py", "audit_pydantic_ai.py",
        "multikey_prose_pilot.py", "summarize_source_sampling.py", "memory_watchdog.py", "mlx_guarded_worker.py")
    study = {"schema": "keyprint.guarded-source-continuation.v1", "study_plan_sha256": digest(public / "plan.json"),
        "runs_sha256": digest(private / "runs.json"), "script_sha256": digest(Path(__file__)),
        "helper_sha256": {n: digest(Path(__file__).with_name(n)) for n in helpers},
        "original_plan": original_plan, "original_progress_sha256": digest(progress_file),
        "original_trace_sha256": {r["review_id"]: digest(original_folder/(r["review_id"]+".json")) for r in previous},
        "incident_sha256": digest(args.incident), "preflight_sha256": digest(args.preflight),
        "preflight_supervisor_sha256": digest(args.supervisor), "cache_limit_bytes": 0,
        "retained_completed_paths": len(previous), "continued_review_ids": [r["review_id"] for r in remaining],
        "selection": "Every remaining original path in original order. The interrupted path is replayed from its original prompt; its prior partial attempt remains retained.",
        "randomness": "Every original draw transcript, no new generation, replacement output or random draws.",
        "scope": "Resource-policy continuation, not a fresh uninterrupted experiment or quality acceptance."}
    save(output / "plan.json", study)
    import keyprint
    from keyprint.experimental.wide_mlx import verify_assets, NFCWideByteLevelBinding
    from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
    from keyprint._engine.legacy._impl.research.token_source_policy import TokenSourceSession
    package = Path(keyprint.__file__).parent
    sources = {str(f.relative_to(package)): digest(f) for f in package.rglob("*.py")}
    if sources != plan["sdk_source_sha256"] or verify_assets(args.model) != identity["model_assets"]:
        raise ValueError("Frozen SDK or model differs")
    for name, expected in identity["dependencies"].items():
        if importlib.metadata.version(name) != expected: raise ValueError("Runtime differs")
    keys = [(args.prior / f"private/key-{i}").read_bytes() for i in range(4)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != plan["key_sha256"]:
        raise ValueError("Recorded keys differ")
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tokenizer = load(str(args.model), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    if set(tokenizer.eos_token_ids) != {248046}: raise ValueError("Runtime EOS differs")
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(), vocabulary_size=248320,
        special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
    if profile.digest != identity["profile_sha256"]: raise ValueError("Profile differs")
    outcomes = list(previous)
    for row in remaining:
        result = {"review_id": row["review_id"], "case": row["case"], "key_slot": row["key_slot"],
            "condition": row["condition"], "matched_steps": 0, "model_calls": 0, "passed": False}
        started, steps, phase = time.monotonic(), [], "receipts"
        session = TokenSourceSession(profile, keys[row["key_slot"]], condition=row["condition"])
        cache = make_prompt_cache(model)
        try:
            folder = private / row["review_id"]
            for name, suffix in (("journal", ".jsonl"), ("report", ".json")):
                if digest(folder / (name + suffix)) != row[name + "_sha256"]: raise ValueError("Receipt differs")
            events, report = audit_journal(folder / "journal.jsonl"), read(folder / "report.json")
            heads = [e for e in events if e["phase"] == "prepared"]
            draws = [e["draw"] for e in events if e["phase"] == "commit_requested"]
            tokens = report["committed_token_ids"]
            if len(heads) != len(draws) or len(draws) != len(tokens): raise ValueError("Step counts differ")
            inputs = events[0]["prompt_token_ids"]
            for head, draw, token in zip(heads, draws, tokens):
                phase = "native_forward"
                result["model_calls"] += 1
                logits = model(mx.array([inputs]), cache=cache)[:, -1, :].astype(mx.float32)
                mx.eval(logits)
                raw = np.array(logits)
                if hashlib.sha256(raw.tobytes()).hexdigest() != head["raw_logits_sha256"]:
                    raise ValueError("Native model-head bytes differ")
                phase = "reference_weights"
                base = ordinary_weights(raw, binding, temperature=plan["settings"]["temperature"], top_k=plan["settings"]["top_k"])
                prepared = session.prepare(base)
                if hashlib.sha256(prepared.probabilities.tobytes()).hexdigest() != head["weights_sha256"]:
                    raise ValueError("Reference weights differ")
                phase = "reference_draw_commit"
                if replay_draw(prepared.probabilities, draw) != token: raise ValueError("Recorded token differs")
                session.commit(prepared, token)
                result["matched_steps"] += 1
                steps.append(distribution_metrics(base, prepared.probabilities, token, binding.eos_ids))
                inputs = [token]
            if session.source_receipt() != report["source_receipt"]: raise ValueError("Source counters differ")
            result["passed"] = True
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error), failed_phase=phase)
        finally:
            cache.clear()
            session.close()
        save(output / (row["review_id"] + ".json"), steps)
        result.update(elapsed_seconds=time.monotonic() - started, diagnostics=aggregate(steps))
        outcomes.append(result)
        save(output / "progress.json", outcomes)
        print(f"Continued to {len(outcomes)}/128; verified {sum(r['passed'] for r in outcomes)}; steps {result['matched_steps']}; elapsed {result['elapsed_seconds']:.1f}s", flush=True)
    if (digest(Path(__file__)) != study["script_sha256"]
            or any(digest(Path(__file__).with_name(n)) != h for n, h in study["helper_sha256"].items())
            or {str(f.relative_to(package)): digest(f) for f in package.rglob("*.py")} != sources):
        raise ValueError("Replay or SDK changed during execution")
    if (digest(progress_file) != study["original_progress_sha256"]
            or any(digest(original_folder/(rid+".json")) != h for rid, h in study["original_trace_sha256"].items())):
        raise ValueError("Original partial replay changed during continuation")
    result_file = output / "result.json"
    save(result_file, {"schema": "keyprint.guarded-source-continuation-results.v1",
        "plan": study, "attempts": len(outcomes), "verified": sum(r["passed"] for r in outcomes),
        "matched_steps": sum(r["matched_steps"] for r in outcomes), "rows": outcomes,
        "quality_acceptance": False, "detector_calibrated": False, "launch_ready": False})
    # The existing complete-cohort validator is unchanged. No successful-only fallback.
    traces = {r["review_id"]: trace_reader(r["review_id"]) for r in previous}
    traces.update({r["review_id"]: read(output/(r["review_id"]+".json")) for r in remaining})
    summary = summarize(rows, outcomes, traces)
    summary["commitment"] = {"results_sha256": digest(result_file),
        "original_trace_sha256": study["original_trace_sha256"],
        "continued_trace_sha256": {r["review_id"]: digest(output/(r["review_id"]+".json")) for r in remaining},
        "resource_policy_changed": True, "retained_completed_paths": len(previous)}
    save(output / "summary.json", summary)



if __name__ == "__main__":
    main()
