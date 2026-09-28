"""Independent native-forward and reference-source replay of all retained runs.

No Keyprint.generate, PortableGeneration, wide projection, sparse source session
or new randomness. Uses direct native forwards, gap-first full-head filtering,
the preserved scalar source policy and exact categorical reference primitives.
The underlying MLX model kernels, tokenizer binding and reference numerical
primitives are shared; this is not an independent implementation of the model.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_pydantic_ai import audit_journal
from multikey_prose_pilot import digest, save


def ordinary_weights(raw, binding, *, temperature, top_k):
    from keyprint._engine.research.keyprint_exact_categorical_v2 import supported_softmax
    if raw.shape != (1, len(binding.pieces)) or raw.dtype != np.float32 or np.isnan(raw).any() or np.isposinf(raw).any():
        raise ValueError("Invalid replay model head")
    row = raw[0].astype(np.float64)
    allowed = np.array([piece is not None or i in binding.eos_ids for i, piece in enumerate(binding.pieces)])
    ids = np.flatnonzero(allowed & np.isfinite(row))
    if not len(ids):
        raise ValueError("Replay support empty")
    losses = np.max(row[ids]) - row[ids]
    possible = losses <= math.nextafter(math.nextafter(600., math.inf) * temperature, math.inf)
    ids, losses = ids[possible], losses[possible] / temperature
    keep = np.isfinite(losses) & (losses <= 600.)
    ids, losses = ids[keep], losses[keep]
    rank = np.lexsort((ids, -row[ids]))[:top_k]
    selected, scaled = ids[rank], -losses[rank]
    order = np.argsort(selected)
    selected, scaled = selected[order], scaled[order]
    probabilities = np.zeros(len(row), np.float64)
    probabilities[selected] = supported_softmax(tuple(map(float, scaled)))
    return probabilities


def replay_draw(probabilities, expected):
    from keyprint._engine.research.keyprint_exact_categorical_v2 import sample_float_weights
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid replay weights")
    selected = np.flatnonzero(probabilities > 0)
    values = iter(expected["transcript"])
    def bits(count):
        entry = next(values)
        if entry["bit_count"] != count:
            raise ValueError("Replay RNG bit request differs")
        return entry["value"]
    draw = sample_float_weights(tuple(map(float, probabilities[selected])), bits)
    if next(values, None) is not None:
        raise ValueError("Unused original draws")
    draw = replace(draw, token_index=int(selected[draw.token_index]))
    if json.loads(json.dumps(asdict(draw))) != expected:
        raise ValueError("Reference selection/transcript differs")
    return draw.token_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    args = p.parse_args()
    public, private = args.root / "public", args.root / "private"
    plan = json.loads((public / "plan.json").read_text())
    identity = json.loads((public / "identity.json").read_text())
    rows = json.loads((private / "runs.json").read_text())
    signature = lambda r: (r["key_slot"], r["case"], r["condition"])
    if len(rows) != 32 or [signature(r) for r in rows] != [signature(r) for r in plan["schedule"]]:
        raise ValueError("Replay requires every original scheduled attempt in order")
    replay_path = private / "native-replay"
    replay_path.mkdir(mode=0o700)
    source = Path(__file__)
    replay_plan = {"schema": "keyprint.wide-native-replay.v1", "study_plan_sha256": digest(public / "plan.json"),
        "runs_sha256": digest(private / "runs.json"), "script_sha256": digest(source),
        "selection": "All 32 original attempts, original order; no selection by outcome, rating or signal",
        "randomness": "Replay every recorded draw; no new random source or generation",
        "independence": "Direct mlx_lm load/model calls; independent gap-first full-width filter; scalar TokenSourceSession; reference softmax/exact categorical primitives. No SDK generation/wide-filter/sparse-source calls. Native kernels, binding and reference primitives shared.",
        "failure_policy": "Retain the first mismatch and consumed work for each attempt; no retries or output replacement"}
    save(public / "native-replay-plan.json", replay_plan)
    import keyprint
    from keyprint.experimental.wide_mlx import verify_assets, NFCWideByteLevelBinding
    from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
    from keyprint._engine.legacy._impl.research.token_source_policy import TokenSourceSession
    for name, expected in plan["sdk_source_sha256"].items():
        if digest(Path(keyprint.__file__).parent / name) != expected:
            raise ValueError("Study source differs")
    if verify_assets(args.model) != identity["model_assets"]:
        raise ValueError("Replay model assets differ")
    import importlib.metadata
    for name, expected in identity["dependencies"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Replay dependencies differ")
    keys = [(args.prior / f"private/key-{i}").read_bytes() for i in range(4)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != plan["key_sha256"]:
        raise ValueError("Replay keys differ")
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tokenizer = load(str(args.model), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    if set(tokenizer.eos_token_ids) != {248046}:
        raise ValueError("Replay runtime EOS differs")
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(), vocabulary_size=248320,
        special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
    if profile.digest != identity["profile_sha256"]:
        raise ValueError("Replay profile differs")
    verified = []
    for row in rows:
        outcome = {"review_id": row["review_id"], "matched_steps": 0, "model_calls": 0, "passed": False}
        start = time.monotonic()
        session = TokenSourceSession(profile, keys[row["key_slot"]], condition=row["condition"])
        cache = make_prompt_cache(model)
        phase = "receipts"
        try:
            folder = private / row["review_id"]
            events = audit_journal(folder / "journal.jsonl")
            report = json.loads((folder / "report.json").read_text())
            if digest(folder / "journal.jsonl") != row["journal_sha256"] or digest(folder / "report.json") != row["report_sha256"]:
                raise ValueError("Replay receipts differ")
            heads = [e for e in events if e["phase"] == "prepared"]
            draws = [e["draw"] for e in events if e["phase"] == "commit_requested"]
            committed = report["committed_token_ids"]
            if len(heads) != len(draws) or len(draws) != len(committed):
                raise ValueError("Replay step counts differ")
            inputs = events[0]["prompt_token_ids"]
            for index, (head, draw, token) in enumerate(zip(heads, draws, committed)):
                phase = "native_forward"
                outcome["model_calls"] += 1
                logits = model(mx.array([inputs]), cache=cache)[:, -1, :].astype(mx.float32)
                mx.eval(logits)
                raw = np.array(logits)
                if hashlib.sha256(raw.tobytes()).hexdigest() != head["raw_logits_sha256"]:
                    raise ValueError("Native model-head bytes differ")
                phase = "reference_weights"
                base = ordinary_weights(raw, binding, temperature=plan["settings"]["temperature"], top_k=plan["settings"]["top_k"])
                prepared = session.prepare(base)
                if hashlib.sha256(prepared.probabilities.tobytes()).hexdigest() != head["weights_sha256"]:
                    raise ValueError("Reference prepared weights differ")
                phase = "reference_draw_commit"
                if replay_draw(prepared.probabilities, draw) != token:
                    raise ValueError("Replay token differs")
                session.commit(prepared, token)
                outcome["matched_steps"] += 1
                inputs = [token]
            if session.source_receipt() != report["source_receipt"]:
                raise ValueError("Source policy/counter receipt differs")
            outcome["passed"] = True
        except Exception as error:
            outcome.update(error_type=type(error).__name__, error=str(error), phase=phase)
        finally:
            cache.clear()
            session.close()
        outcome["elapsed_seconds"] = time.monotonic() - start
        verified.append(outcome)
        save(replay_path / "progress.json", verified)
        print(f"Replayed {len(verified)}/32; passed {sum(r['passed'] for r in verified)}; steps {outcome['matched_steps']}; elapsed {outcome['elapsed_seconds']:.1f}s", flush=True)
    if digest(source) != replay_plan["script_sha256"]:
        raise ValueError("Replay implementation changed during execution")
    result = {"schema": "keyprint.wide-native-replay-results.v1", "plan": replay_plan,
        "attempts": len(verified), "passed": sum(r["passed"] for r in verified),
        "matched_steps": sum(r["matched_steps"] for r in verified),
        "model_calls": sum(r["model_calls"] for r in verified), "rows": verified,
        "quality_acceptance": False, "detector_calibrated": False, "launch_ready": False}
    save(public / "native-replay-results.json", result)


if __name__ == "__main__":
    main()
