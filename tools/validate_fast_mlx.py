"""Compare reference/experimental execution on actual pinned Qwen model heads.

Each sampled head is supplied to both pipelines with identical public-fixture
random streams. A mismatch stops that attempt; no replacement or retry. This
checks arithmetic/token-path parity, not private-key security or serving speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from benchmark_serving import require_storage, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_storage(args.output)
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from keyprint.backends.mlx import MLXModel
    from keyprint.experimental import fast_mlx
    from keyprint._engine.research.keyprint_candidate_v3 import Candidate
    from keyprint._engine.research.keyprint_candidate_v3_caller import DurableJournal
    cases_path = Path(__file__).with_name("inference_cases.json")
    cases = json.loads(cases_path.read_text())
    key = bytes(range(32))
    plan = {"cases": cases, "cases_sha256": sha(cases_path), "script_sha256": sha(Path(__file__)),
            "execution_sha256": sha(Path(fast_mlx.__file__)), "key": "Public fixture bytes 0..31",
            "randomness": "Two independent Random(20260920 + case index) instances per pair; exact draw records compared",
            "scope": "Actual model heads and full paired arithmetic/token-path parity; no serving or detection acceptance",
            "failure_rule": "Retain every attempt; stop mismatched attempt without replacement"}
    (public / "plan.json").write_text(json.dumps(plan, indent=2))
    backend = MLXModel.load(args.model)
    candidate = Candidate()
    rows = []
    for index, case in enumerate(cases):
        for condition in ("ordinary", "marked"):
            name = f"{case['id']}-{condition}"
            directory = args.output / name
            directory.mkdir()
            reference = candidate.pipeline(key, condition=condition)
            optimized = fast_mlx.pipeline(key, condition=condition)
            left, right = random.Random(20260920 + index), random.Random(20260920 + index)
            row = {"case": case["id"], "condition": condition, "max_tokens": case["max_tokens"]}
            heads, error = [], None
            try:
                prompt = backend.encode_prompt(case["prompt"])
                cache = make_prompt_cache(backend.model)
                if len(prompt) > 1:
                    backend.model(mx.array([prompt[:-1]], dtype=mx.int32), cache=cache)
                    mx.eval([entry.state for entry in cache])
                next_ids = [prompt[-1]]
                with DurableJournal(directory / "parity.jsonl") as journal:
                    journal.append({"phase": "start", "prompt_ids": prompt,
                                    "reference_identity": reference.receipt()["runtime"],
                                    "optimized_identity": optimized.receipt()["runtime"]})
                    for step in range(case["max_tokens"]):
                        output = backend.model(mx.array([next_ids], dtype=mx.int32), cache=cache)
                        raw = output[:, -1, :].astype(mx.float32)
                        mx.eval(raw)
                        raw = np.array(raw)
                        digest = hashlib.sha256(raw.tobytes()).hexdigest()
                        journal.append({"phase": "before_sample", "index": step, "raw_head_sha256": digest})
                        a = reference.step(raw, left.getrandbits)
                        b = optimized.step(raw, right.getrandbits)
                        if (a.token_id != b.token_id or a.emitted_text != b.emitted_text
                                or (a.stopped is None) != (b.stopped is None)
                                or reference._raw.sampling_records[-1] != optimized._raw.sampling_records[-1]):
                            raise ValueError("Token, probability hash, draw record or rendered increment differs")
                        heads.append(digest)
                        journal.append({"phase": "matched_commit", "index": step, "token_id": a.token_id})
                        if a.stopped is not None:
                            break
                        next_ids = [a.token_id]
                    a, b = reference.finish(), optimized.finish()
                    if (a.visible.text != b.visible.text or a.reason != b.reason
                            or a.committed_token_ids != b.committed_token_ids
                            or a.visible.generation_score != b.visible.generation_score
                            or a.visible.text_score != b.visible.text_score):
                        raise ValueError("Final text, token path or diagnostic counts differ")
                    row.update(text=a.visible.text, tokens=len(a.committed_token_ids), reason=a.reason,
                               text_sha256=hashlib.sha256(a.visible.text.encode()).hexdigest())
                    journal.append({"phase": "complete", "tokens": row["tokens"], "text_sha256": row["text_sha256"]})
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                reference.close(); optimized.close()
                for label, item in (("reference", reference), ("optimized", optimized)):
                    (directory / (label + ".json")).write_text(json.dumps(item.receipt(), indent=2, ensure_ascii=False))
            row.update(error=error, matched_steps=len(heads), raw_head_hashes=heads,
                       reference_sha256=sha(directory / "reference.json"), optimized_sha256=sha(directory / "optimized.json"),
                       hmac_contexts_cleared=all(not c.session.profile._state for c in optimized._raw.carriers))
            if not row["hmac_contexts_cleared"]:
                row["error"] = {"type": "UnclearedContext", "message": "HMAC context remained after close"}
            (public / (name + ".json")).write_text(json.dumps(row, indent=2, ensure_ascii=False))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("text", "raw_head_hashes")}), flush=True)
    summary = {"status": "pass" if len(rows) == 12 and not any(r["error"] for r in rows) else "failed",
               "outputs": len(rows), "errors": sum(r["error"] is not None for r in rows),
               "matched_model_steps": sum(r["matched_steps"] for r in rows),
               "default_sdk_changed": False, "serving_accepted": False, "scope": plan["scope"]}
    (public / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return int(summary["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
