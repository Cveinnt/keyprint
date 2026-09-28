"""Replay every source-grounded path and measure conditional sampling changes.

No new randomness, output replacement or SDK changes. Measurements compare the
base and marked distributions at the same recorded prefix, not counterfactual
future text or a causal estimate of semantic harm.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_pydantic_ai import audit_journal
from multikey_prose_pilot import digest, save
from replay_wide_mlx import ordinary_weights, replay_draw


def distribution_metrics(base, effective, token, eos_ids):
    if (base.shape != effective.shape or base.ndim != 1 or base.dtype != np.float64
            or effective.dtype != np.float64 or type(token) is not int or not 0 <= token < len(base)
            or not np.isfinite(base).all() or not np.isfinite(effective).all()
            or (base < 0).any() or (effective < 0).any()):
        raise ValueError("Invalid diagnostic distribution")
    ids = np.flatnonzero((base > 0) | (effective > 0))
    if not len(ids) or (base[ids] <= 0).any() or base[token] <= 0 or effective[token] <= 0:
        raise ValueError("Changed support or impossible recorded token")
    p, q = base[ids], effective[ids]
    # Normalization here is telemetry only; replay uses untouched reference
    # weights and original exact random-bit transcripts.
    psum, qsum = math.fsum(p), math.fsum(q)
    if not math.isfinite(psum) or not math.isfinite(qsum) or min(psum, qsum) <= 0:
        raise ValueError("Invalid distribution mass")
    p, q = p / psum, q / qsum
    positive = q > 0
    p_token, q_token = float(base[token] / psum), float(effective[token] / qsum)
    eos = np.isin(ids, list(eos_ids))
    return {"base_max": float(p.max()), "effective_max": float(q.max()),
        "base_entropy_bits": float(-np.sum(p * np.log2(p))),
        "effective_entropy_bits": float(-np.sum(q[positive] * np.log2(q[positive]))),
        "kl_effective_to_base_nats": max(0., float(np.sum(q[positive] * (np.log(q[positive]) - np.log(p[positive]))))),
        "total_variation": float(np.sum(np.abs(q - p)) / 2),
        "selected_base_probability": p_token, "selected_effective_probability": q_token,
        "selected_base_rank": 1 + int(np.sum(p > p_token)),
        "eos_base_mass": float(np.sum(p[eos])), "eos_effective_mass": float(np.sum(q[eos])),
        "selected_is_eos": token in eos_ids, "support": len(ids)}


def aggregate(steps):
    if not steps: return {"steps": 0}
    numeric = [k for k in steps[0] if k not in {"selected_is_eos"}]
    return {"steps": len(steps), "mean": {k: math.fsum(r[k] for r in steps) / len(steps) for k in numeric},
        "selected_base_below_1pct": sum(r["selected_base_probability"] < .01 for r in steps),
        "selected_base_below_01pct": sum(r["selected_base_probability"] < .001 for r in steps),
        "effective_above_99pct": sum(r["effective_max"] > .99 for r in steps),
        "newly_above_99pct": sum(r["effective_max"] > .99 and r["base_max"] <= .99 for r in steps),
        "selected_eos": [r for r in steps if r["selected_is_eos"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    args = p.parse_args()
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
    output = private / "sampling-diagnosis"
    output.mkdir(mode=0o700)  # fail if this logical replay already exists
    helpers = ("replay_wide_mlx.py", "audit_pydantic_ai.py", "multikey_prose_pilot.py")
    study = {"schema": "keyprint.source-sampling-diagnosis.v1", "study_plan_sha256": digest(public / "plan.json"),
        "runs_sha256": digest(private / "runs.json"), "script_sha256": digest(Path(__file__)),
        "helper_sha256": {n: digest(Path(__file__).with_name(n)) for n in helpers},
        "selection": "All 128 recorded paths and all steps, original order, regardless of rating, key or signal.",
        "hypothesis": "Measure whether marking concentrates conditional distributions, amplifies low-base-probability choices or changes EOS probability. Coverage loss may have other causes; these measurements do not establish semantic causation.",
        "randomness": "Use every original draw transcript; no new generation, random draws, retries or repaired output.",
        "independence": "Direct native forwards, gap-first full-head reference filter, scalar TokenSourceSession and exact categorical reference. Shared kernels, tokenizer binding and reference primitives.",
        "scope": "Exploratory diagnosis after frozen factual review. No candidate promotion, old-profile detector threshold, causal quality claim or launch acceptance."}
    save(public / "sampling-diagnosis-plan.json", study)
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
    outcomes = []
    for row in rows:
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
        print(f"Diagnosed {len(outcomes)}/128; verified {sum(r['passed'] for r in outcomes)}; steps {result['matched_steps']}; elapsed {result['elapsed_seconds']:.1f}s", flush=True)
    if (digest(Path(__file__)) != study["script_sha256"]
            or any(digest(Path(__file__).with_name(n)) != h for n, h in study["helper_sha256"].items())
            or {str(f.relative_to(package)): digest(f) for f in package.rglob("*.py")} != sources):
        raise ValueError("Replay or SDK changed during execution")
    save(public / "sampling-diagnosis-results.json", {"schema": "keyprint.source-sampling-diagnosis-results.v1",
        "plan": study, "attempts": len(outcomes), "verified": sum(r["passed"] for r in outcomes),
        "matched_steps": sum(r["matched_steps"] for r in outcomes), "rows": outcomes,
        "quality_acceptance": False, "detector_calibrated": False, "launch_ready": False})


if __name__ == "__main__":
    main()
