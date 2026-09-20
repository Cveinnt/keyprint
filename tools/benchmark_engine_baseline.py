"""Unmodified MLX-LM versus complete ordinary/marked Keyprint generation.

Single requests on one loaded model; not an HTTP, batching or production test.
The paths intentionally retain their native sampling/rendering/bookkeeping
semantics. Ratios measure complete developer-facing paths, not pure marking.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import time

from benchmark_serving import REPEATS, require_storage, sha, summarize

ARMS = ("engine", "sdk_ordinary", "sdk_marked")


def order(repeat, index):
    offset = (repeat + index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def collect_engine(stream, *, cap, eos_ids):
    """Keep every upstream yielded token, including the terminal EOS or cap."""
    tokens, segments, terminal = [], [], None
    for item in stream:
        if terminal is not None:
            raise ValueError("Engine yielded after its terminal response")
        tokens.append(int(item.token))
        segments.append(item.text)
        if item.generation_tokens != len(tokens) or len(tokens) > cap:
            raise ValueError("Engine count differs from its yielded token path")
        if item.finish_reason is not None:
            terminal = item.finish_reason
    if terminal not in ("stop", "length"):
        raise ValueError("Engine did not finish explicitly")
    if (terminal == "stop") != (tokens[-1] in eos_ids):
        raise ValueError("Engine finish reason differs from terminal token")
    if terminal == "length" and len(tokens) != cap:
        raise ValueError("Engine length finish did not reach the declared cap")
    return {"token_ids": tokens, "text": "".join(segments),
            "completion_tokens": len(tokens),
            "completion": "eos" if terminal == "stop" else "length"}


def analyze(rows, cases):
    expected = {(case, repeat, arm) for case in cases
                for repeat in range(REPEATS) for arm in ARMS}
    actual = {(r["case"], r["repeat"], r["arm"]) for r in rows}
    if len(rows) != len(expected) or actual != expected or any("error" in r for r in rows):
        raise ValueError("Every declared arm required exactly once, without errors")
    result = {}
    for baseline, tested in (("engine", "sdk_ordinary"), ("engine", "sdk_marked"),
                             ("sdk_ordinary", "sdk_marked")):
        pair = [{**r, "condition": "ordinary" if r["arm"] == baseline else "marked"}
                for r in rows if r["arm"] in (baseline, tested)]
        stats = summarize(pair, cases)
        result[f"{tested}_over_{baseline}"] = {
            "seconds_per_committed_token": stats["seconds_per_committed_token"],
            "request_latency": stats["request_latency"],
            "local_5pct_screen_pass": stats["incremental_5pct_timing_screen_pass"],
        }
    return {"comparisons": result, "production_overhead_accepted": False,
            "scope": "Fixed-workload developer-path comparison; different sampling laws and retained output lengths; no batching or HTTP"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-host-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.idle_host_confirmed:
        parser.error("finish other inference and benchmarks before measurement")
    free = require_storage(args.output)
    cases_path = Path(__file__).with_name("inference_cases.json")
    cases = json.loads(cases_path.read_text())
    if len(cases) != 6 or len({c["id"] for c in cases}) != 6:
        raise ValueError("All six frozen comparison prompts required")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    def write(path, data):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from keyprint import Keyprint
    plan = {"scope": __doc__, "script_sha256": sha(Path(__file__)),
            "analysis_sha256": sha(Path(__file__).with_name("benchmark_serving.py")),
            "cases_sha256": sha(cases_path), "cases": cases, "arms": ARMS,
            "repeats": REPEATS, "measured_requests": 72,
            "warmups": "One retained 32-token request per arm on first case, excluded prospectively",
            "order": "Repeat outer, case inner; rotate three arms left by repeat + case index modulo three",
            "engine": "Unmodified mlx_lm.stream_generate, make_sampler(temp=0.7, top_k=100), no extra logits processors; fresh default cache",
            "sdk": "experimental-native; ordinary and marked; temperature=0.7, top_k=100; fresh response cache",
            "differences": "Engine uses its own device sampling and special-token policy. Keyprint uses its bound CPU binary64 law, channel policy, crypto draws and durable journals/reports. Equal parameter names do not imply identical distributions.",
            "randomness": "Engine fixed independent per-request seeds; Keyprint independent cryptographic draws; no identical-path or quality claim",
            "token_denominator": "Actual returned/committed tokens including EOS, counted in both paths",
            "timed_scope": "Prompt encoding, model generation, native rendering, response collection; Keyprint also writes its durable reports/journals inside generate",
            "excluded_scope": "Imports, model loading, benchmark artifact writing, inspection, HTTP, batching",
            "primary": "SDK marked over unmodified engine seconds/token; geometric mean and one-sided 95% paired bootstrap upper; local screen <=1.05",
            "secondary": "SDK ordinary/engine and marked/ordinary; request latency ratios",
            "failure_rule": "Retain all warmups, failures and caps; no retries, replacements or outlier exclusions; failures prevent a completed summary",
            "platform": platform.platform(), "python": platform.python_version(),
            "free_bytes_before": free, "os_isolation": False,
            "versions": {p: importlib.metadata.version(p) for p in ("keyprint", "keyprint-native", "mlx", "mlx-lm", "numpy")},
            "engine_source_sha256": {str(Path(inspect.getfile(f)).name): sha(Path(inspect.getfile(f)))
                                     for f in (stream_generate, make_sampler, TokenizerWrapper)}}
    write(public / "plan.json", plan)
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(key)
    rows, warmups, failure, analysis = [], [], None, None
    try:
        start = time.perf_counter()
        candidate = Keyprint.from_mlx(args.model, key=key, execution="experimental-native")
        backend = candidate._backend
        mx.synchronize()
        write(public / "load.json", {"seconds": time.perf_counter() - start,
                                      "identity": candidate.identity, "model_path": str(args.model.resolve())})
        def attempt(case, repeat, arm, *, warmup=False):
            cap = 32 if warmup else case["max_tokens"]
            name = f"{'warmup-' if warmup else ''}{case['id']}-{repeat}-{arm}"
            row = {"id": name, "case": case["id"], "repeat": repeat, "arm": arm, "max_tokens": cap}
            seed = 20260920 + (repeat + 1) * len(cases) + cases.index(case)
            mx.random.seed(seed)
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            try:
                if arm == "engine":
                    prompt_ids = backend.encode_prompt(case["prompt"])
                    result = collect_engine(stream_generate(backend.model, backend.tokenizer,
                        prompt_ids, max_tokens=cap, sampler=make_sampler(temp=.7, top_k=100)),
                        cap=cap, eos_ids=backend.tokenizer.eos_token_ids)
                    mx.synchronize()
                    row["seconds"] = time.perf_counter() - started
                    row.update(result, prompt_ids=prompt_ids, seed=seed)
                    row["decoded_text"] = backend.tokenizer.decode(result["token_ids"], skip_special_tokens=True)
                    row["native_rendering_matches_decode"] = row["text"] == row["decoded_text"]
                else:
                    condition = arm.removeprefix("sdk_")
                    result = candidate.generate(case["prompt"], condition=condition,
                                                max_tokens=cap, output=args.output / name)
                    mx.synchronize()
                    row["seconds"] = time.perf_counter() - started
                    payload = result.report["payload"]
                    row.update(text=result.text, token_ids=payload["committed_token_ids"],
                               condition=condition, completion=payload["completion"],
                               completion_tokens=result.report["usage"]["completion_tokens"],
                               report_sha256=sha(args.output / name / "report.json"),
                               journal_sha256=sha(args.output / name / "journal.jsonl"))
                if row["completion_tokens"] != len(row["token_ids"]):
                    raise ValueError("Token denominator differs from actual path")
                row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            except Exception as exc:
                row.update(seconds=time.perf_counter() - started,
                           error={"type": type(exc).__name__, "message": str(exc)})
            row["peak_bytes"] = mx.get_peak_memory()
            write(public / (name + ".json"), row)
            print(json.dumps({k: row[k] for k in ("id", "seconds", "completion_tokens", "completion", "error") if k in row}), flush=True)
            return row
        for arm in ARMS:
            warmups.append(attempt(cases[0], -1, arm, warmup=True))
        if any("error" in row for row in warmups):
            raise ValueError("Warmup failure prevents measurement")
        for repeat in range(REPEATS):
            for index, case in enumerate(cases):
                for arm in order(repeat, index):
                    rows.append(attempt(case, repeat, arm))
        analysis = analyze(rows, [c["id"] for c in cases])
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    summary = {"status": "completed" if failure is None else "incomplete",
               "failure": failure, "requests": len(rows), "warmups": len(warmups),
               "analysis": analysis, "production_overhead_accepted": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())
