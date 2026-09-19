"""Diagnose already-opened MLX paths using exact saved randomness and hashes.

No new random draws, detector training or threshold selection. Requires private
generation artifacts and the original prompt/model. This is an oracle diagnostic,
not a pasted-text detector. Only aggregate public/ results may be exported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_journal(path):
    previous, events = "0" * 64, []
    for index, line in enumerate(Path(path).read_bytes().splitlines(keepends=True)):
        row = json.loads(line)
        if not line.endswith(b"\n") or row["sequence"] != index or row["previous_sha256"] != previous:
            raise ValueError("Broken source journal chain")
        previous = hashlib.sha256(line).hexdigest()
        events.append(row["event"])
    return events


def metrics(base, marked, centered, selected):
    """Moments on the exact positive token support, with binary64 log ratios."""
    if len(base) != len(marked) or len(base) != len(centered) or not 0 <= selected < len(base):
        raise ValueError("Mismatched probability support")
    if any(not math.isfinite(v) or v <= 0 for v in [*base, *marked]):
        raise ValueError("Only positive finite support is allowed")
    q_total, p_total = math.fsum(base), math.fsum(marked)
    if abs(q_total-1) > 1e-12 or abs(p_total-1) > 1e-12:
        raise ValueError("Probability normalization differs")
    q, p = [v/q_total for v in base], [v/p_total for v in marked]
    mean_q = math.fsum(a*b for a,b in zip(q,centered))
    mean_p = math.fsum(a*b for a,b in zip(p,centered))
    return {"base_entropy_bits": -math.fsum(v*math.log2(v) for v in q),
            "base_max_probability": max(q),
            "expected_base_centered_bits": mean_q,
            "expected_marked_centered_bits": mean_p,
            "expected_bit_lift": mean_p-mean_q,
            "base_score_variance": math.fsum(v*(s-mean_q)**2 for v,s in zip(q,centered)),
            "marked_kl_base_bits": math.fsum(v*(math.log2(v)-math.log2(w)) for v,w in zip(p,q)),
            "total_variation": math.fsum(abs(v-w) for v,w in zip(p,q))/2,
            "observed_centered_bits": centered[selected],
            "selected_log2_ratio": math.log2(p[selected])-math.log2(q[selected])}


def replay(backend, candidate, root, plan, name, key, sink):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from keyprint._engine.legacy._impl.research.token_source_policy import transform
    saved = json.loads((root/name/"report.json").read_text())["report"]
    public = json.loads((root/"public"/(name+".json")).read_text())
    records = saved["payload"]["sampling_records"]
    tokens = saved["payload"]["committed_token_ids"]
    events = read_journal(root/name/"journal.jsonl")
    start = next(e for e in events if e["kind"] == "response_started")
    heads = [e for e in events if e["kind"] == "prepared_step"]
    draws = iter(e for e in events if e["kind"] == "random_bits_returned")
    prompt = backend.encode_prompt(public["prompt"] + plan["prompt_suffix"])
    packed = json.dumps(prompt,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()
    if hashlib.sha256(packed).hexdigest() != start["prompt_sha256"]:
        raise ValueError("Original prompt mismatch")
    if candidate.identity["runtime_profile_sha256"] != saved["target_identity"]["runtime_profile_sha256"]:
        raise ValueError("Original sampler identity mismatch")
    if len(records) != len(tokens) or len(heads) != len(tokens):
        raise ValueError("Incomplete saved path")
    pipeline = candidate.pipeline(key, condition=public["condition"])
    session = pipeline._raw._current.session
    prepare = session.prepare
    measured = []
    pending = {}
    def observe(base):
        prepared = prepare(base)
        context = session._context
        fresh = context not in session._used
        marked = prepared.probabilities if public["condition"] == "marked" else (
            transform(base,session.profile,key,context,(),{"branch_roundups":0,"partition_roundups":0})
            if fresh else base.copy())
        support = np.flatnonzero(base > 0)
        scores = []
        for token in support:
            label = session.profile.classes[int(token)]
            scores.append(sum(session.profile.bits(key,context,label))-15 if fresh and label else 0)
        pending.update(base=base[support].tolist(), marked=marked[support].tolist(),
                       scores=scores, support=support.tolist(), fresh=fresh)
        return prepared
    # Instance-local observation only. Original preparation result is returned
    # unchanged; its distribution and sampled tokens are checked against receipts.
    session.prepare = observe
    cache = make_prompt_cache(backend.model)
    ids = mx.array(prompt,dtype=mx.int32)
    offset = 0
    while len(ids)-offset > 1:
        count = min(start["prefill_step_size"],len(ids)-offset-1)
        backend.model(ids[None,offset:offset+count],cache=cache)
        mx.eval([entry.state for entry in cache])
        offset += count
    next_input = ids[offset:]
    replayed_draws = 0
    for index, token in enumerate(tokens):
        output = backend.model(next_input[None],cache=cache)
        head = output[:,-1,:].astype(mx.float32)
        mx.eval(head)
        raw = np.array(head)
        if hashlib.sha256(raw.tobytes()).hexdigest() != heads[index]["raw_logits_sha256"]:
            raise ValueError(f"Raw logits differ at step {index}")
        def saved_bits(count):
            nonlocal replayed_draws
            draw = next(draws)
            if draw["bits"] != count or draw["sample_index"] != index:
                raise ValueError("Randomness replay request differs")
            replayed_draws += 1
            return int(draw["value_decimal"])
        step = pipeline.step(raw,saved_bits)
        if step.token_id != token or pipeline._raw.sampling_records[-1] != records[index]:
            raise ValueError(f"Original token/probability/random transcript differs at step {index}")
        row = {"index":index,"token_id":token,"fresh_context":pending["fresh"],
               **metrics(pending["base"],pending["marked"],pending["scores"],pending["support"].index(token))}
        measured.append(row)
        sink.write(json.dumps(row,allow_nan=False)+"\n"); sink.flush()
        next_input = mx.array([token],dtype=mx.int32)
    if next(draws,None) is not None:
        raise ValueError("Unconsumed saved randomness")
    final = pipeline.finish() if not pipeline.receipt()["finalized"] else pipeline.receipt()["final"]
    # Receipt final is a plain mapping; finish returns a carrier dataclass.
    text = final["visible"]["text"] if isinstance(final,dict) else final.visible.text
    if text != public["text"]:
        raise ValueError("Replay text differs")
    sums = {k:math.fsum(r[k] for r in measured) for k in (
        "expected_base_centered_bits","expected_marked_centered_bits","expected_bit_lift",
        "marked_kl_base_bits","observed_centered_bits","selected_log2_ratio")}
    bins = []
    for low, high in ((0,.1),(.1,.5),(.5,1),(1,2),(2,100)):
        selected = [r for r in measured if low <= r["base_entropy_bits"] < high]
        bins.append({"entropy_range_bits":[low,high],"steps":len(selected),
                     "expected_bit_lift":math.fsum(r["expected_bit_lift"] for r in selected),
                     "marked_kl_base_bits":math.fsum(r["marked_kl_base_bits"] for r in selected)})
    return {"id":name,"steps":len(measured),"new_random_draws":0,"replayed_draws":replayed_draws,
            "all_raw_logits_distributions_tokens_draws_match":True,
            "base_entropy_mean_bits":math.fsum(r["base_entropy_bits"] for r in measured)/len(measured),
            "base_max_probability_over_90pct":sum(r["base_max_probability"]>.9 for r in measured),
            "total_variation_mean":math.fsum(r["total_variation"] for r in measured)/len(measured),
            "sums":sums,"entropy_bins":bins,"original_matching_statistic":public["scores"]["matching"]["statistic"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",nargs="+",required=True)
    args=parser.parse_args()
    # Only known study rows are permitted, never traversal outside the study.
    known={p.stem for p in (args.study/"public").glob("heldout-*.json")}
    if len(args.cases)!=len(set(args.cases)) or not set(args.cases)<=known:
        raise ValueError("Select distinct existing held-out case IDs")
    args.output.mkdir(mode=0o700); public=args.output/"public";public.mkdir()
    plan=json.loads((args.study/"public/plan.json").read_text())
    declaration={"scope":"Opened-data oracle capacity diagnosis; not a calibrated detector or fresh confirmation",
                 "script_sha256":sha(__file__),"cases":args.cases,"study_plan_sha256":sha(args.study/"public/plan.json"),
                 "source_reports":{n:sha(args.study/n/"report.json") for n in args.cases},
                 "source_journals":{n:sha(args.study/n/"journal.jsonl") for n in args.cases}}
    (public/"plan.json").write_text(json.dumps(declaration,indent=2))
    results=[];error=None;started=time.monotonic()
    try:
        from keyprint import Keyprint
        from keyprint._engine.research.keyprint_candidate_v3 import Candidate
        key=(args.study/"owner.key").read_bytes()
        loaded=Keyprint.from_mlx(args.model,key=key)
        candidate=Candidate(temperature=plan["temperature"],top_k=plan["top_k"])
        for name in args.cases:
            with (args.output/(name+".steps.jsonl")).open("x") as sink:
                row=replay(loaded._backend,candidate,args.study,plan,name,key,sink)
            results.append(row)
            (public/(name+".json")).write_text(json.dumps(row,indent=2))
            print(json.dumps(row),flush=True)
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc)}
    (public/"summary.json").write_text(json.dumps({"results":results,"error":error,
        "status":"pass" if error is None else "failed","seconds":time.monotonic()-started,
        "deployment_calibrated":False,"scope":declaration["scope"]},indent=2))
    return int(error is not None)


if __name__=="__main__":
    raise SystemExit(main())
