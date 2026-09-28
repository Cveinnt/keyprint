"""Measure ordinary newer-model fidelity before implementing a watermark adapter.

This is upstream MLX-LM inference, not a Keyprint-supported backend or detector.
Keep every attempt, raw token and output; never repair text or retry a failure.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import uuid

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"
REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def schedule(cases):
    return [dict(case=case["id"], repetition=rep,
                 seed=2026092801 + index * 2 + rep)
            for rep in range(2) for index, case in enumerate(cases)]


def reconcile(events, decoded, eos_ids, cap):
    """Do not drop content tokens, repair whitespace or hide cap termination."""
    if not events or any(e["finish_reason"] is not None for e in events[:-1]):
        raise ValueError("Require one complete generation event sequence")
    final = events[-1]
    ids = [e["token"] for e in events]
    stop = final["finish_reason"] == "stop"
    if stop:
        if ids[-1] not in eos_ids or any(i in eos_ids for i in ids[:-1]):
            raise ValueError("EOS does not match the terminal event")
        visible_ids = ids[:-1]
    else:
        if final["finish_reason"] != "length" or len(ids) != cap or any(i in eos_ids for i in ids):
            raise ValueError("Unexpected termination")
        visible_ids = ids
    if final["generation_tokens"] != len(ids):
        raise ValueError("Token count differs from retained events")
    text = "".join(e["text"] for e in events)
    if text != decoded(visible_ids):
        raise ValueError("Stream differs from native token decoding")
    return {"text": text, "sampled_token_ids": ids,
            "visible_token_ids": visible_ids, "tokens": len(ids),
            "completion": "eos" if stop else "length"}


class ObservedModel:
    """Count real upstream forwards without changing their arguments or logits."""
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model.name != REVISION or not args.model.is_dir():
        parser.error("Use the downloaded pinned Qwen3.5-9B snapshot")
    cases_path = Path(__file__).with_name("long_fidelity_cases.json")
    cases = json.loads(cases_path.read_text())
    if len(cases) != 4 or len({c["id"] for c in cases}) != 4:
        raise ValueError("All four unchanged long-form tasks required")
    args.output.mkdir(mode=0o700)
    public, private = args.output / "public", args.output / "private"
    public.mkdir(); private.mkdir(mode=0o700)
    assets = {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}
    plan = {"schema": "keyprint.model-baseline.v1", "model": MODEL_ID,
            "revision": REVISION, "assets_sha256": assets,
            "script_sha256": digest(Path(__file__)), "cases_sha256": digest(cases_path),
            "cases": cases, "schedule": schedule(cases),
            "settings": {"temperature": .7, "top_k": 100, "top_p": 0.0,
                         "max_tokens": 1024, "enable_thinking": False},
            "dependencies": {n: importlib.metadata.version(n) for n in
                             ("mlx", "mlx-lm", "transformers", "tokenizers", "numpy")},
            "failure_policy": "Eight attempts, two per unchanged task. No retries, replacements, repairs or output selection. Retain all errors/caps.",
            "review": "Assistant fact/no-addition/language/prose labels frozen before opening seed/order metadata. Model identity and source tasks are known; not independent or model-blind review.",
            "decision": "Model feasibility only: report content, language, format and execution separately and jointly, with every failure retained. This is not a new launch gate. Determine whether the prior all-failing baseline persists before attributing harm to a watermark; native binding and paired watermark evidence must be qualified separately. Even eight passes cannot prove general preservation.",
            "scope": "Upstream ordinary text-only MLX-LM, not Keyprint generation. No watermark keys, detector scores, private-algorithm identification or launch acceptance. Same opened source tasks; different model/runtime sampling is not a controlled causal comparison to prior SDK samples."}
    save(public / "plan.json", plan)
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    try:
        model, tokenizer = load(str(args.model), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
        model = ObservedModel(model)
    except Exception as error:
        save(private / "setup-error.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
    rows = []
    by_case = {c["id"]: c for c in cases}
    for attempt in plan["schedule"]:
        row = dict(attempt, review_id=uuid.uuid4().hex[:12])
        folder = private / row["review_id"]
        folder.mkdir()
        events = []
        start = time.monotonic()
        before = model.calls
        try:
            mx.random.seed(row["seed"])
            state_before = [x.tolist() for x in mx.random.state]
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": by_case[row["case"]]["prompt"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            save(folder / "prompt.json", {"ids": prompt_ids, "seed": row["seed"], "random_state_before": state_before})
            for event in stream_generate(model, tokenizer, prompt_ids, max_tokens=1024,
                                         sampler=make_sampler(temp=.7, top_k=100)):
                events.append({"text": event.text, "token": int(event.token),
                               "generation_tokens": event.generation_tokens,
                               "finish_reason": event.finish_reason,
                               "selected_logprob": float(event.logprobs[event.token].item())})
            decoded = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            row.update(reconcile(events, decoded, tokenizer.eos_token_ids, 1024))
            row["random_state_after"] = [x.tolist() for x in mx.random.state]
            row["prompt_tokens"] = len(prompt_ids)
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error),
                       text="".join(e["text"] for e in events))
        row.update(model_forward_calls=model.calls-before, seconds=time.monotonic()-start)
        save(folder / "events.json", events)
        row["events_sha256"] = digest(folder / "events.json")
        rows.append(row)
        save(private / "runs.json", rows)
        print(f"Completed {len(rows)}/8; errors: {sum('error_type' in r for r in rows)}", flush=True)
    review = [dict(review_id=r["review_id"], case=by_case[r["case"]], text=r.get("text", ""),
                   completion=r.get("completion"), error_type=r.get("error_type")) for r in rows]
    random.SystemRandom().shuffle(review)
    save(public / "blind-review.json", review)
    print("Baseline review ready; no watermark integration or acceptance claimed.", flush=True)


if __name__ == "__main__":
    main()
