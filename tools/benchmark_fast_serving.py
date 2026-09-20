"""Interleaved reference/experimental MLX serving comparison on one loaded model.

All four execution/condition cells are observed for each fixed case/repeat.
No native-server, semantic-quality or production acceptance is inferred.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from benchmark_serving import (
    BOOTSTRAPS, BOOTSTRAP_SEED, MIN_FREE_BYTES, REPEATS, require_storage, sha, summarize,
)

CELLS = (("reference", "ordinary"), ("fast", "ordinary"),
         ("fast", "marked"), ("reference", "marked"))


def cell_order(repeat, index):
    offset = (repeat + index) % 4
    return CELLS[offset:] + CELLS[:offset]


def analyze(rows, cases):
    expected = {(e, c, case, rep) for e, c in CELLS for case in cases for rep in range(REPEATS)}
    observed = {(r["execution"], r["condition"], r["case"], r["repeat"]): r for r in rows}
    if len(rows) != len(expected) or set(observed) != expected:
        raise ValueError("Every declared execution/condition cell required exactly once")
    by_execution = {e: summarize([r for r in rows if r["execution"] == e], cases)
                    for e in ("reference", "fast")}
    comparisons = {}
    for condition in ("ordinary", "marked"):
        groups = [[math.log(
            (observed["fast", condition, case, rep]["seconds"] / observed["fast", condition, case, rep]["completion_tokens"]) /
            (observed["reference", condition, case, rep]["seconds"] / observed["reference", condition, case, rep]["completion_tokens"]))
                   for rep in range(REPEATS)] for case in cases]
        values = np.asarray(groups)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, REPEATS, size=(BOOTSTRAPS, len(cases), REPEATS))
        boot = values[np.arange(len(cases))[None, :, None], indices].mean(axis=(1, 2))
        comparisons[condition] = {
            "geometric_mean_seconds_per_token_ratio": float(np.exp(values.mean())),
            "one_sided_95_upper_ratio": float(np.exp(np.quantile(boot, .95))),
        }
    return {"within_execution": by_execution, "fast_over_reference": comparisons,
            "primary_fast_5pct_screen_pass": by_execution["fast"]["incremental_5pct_timing_screen_pass"],
            "native_server_overhead_measured": False, "production_overhead_accepted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-host-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.idle_host_confirmed:
        raise ValueError("Stop other inference/benchmark workloads before measurement")
    free = require_storage(args.output)
    cases_path = Path(__file__).with_name("inference_cases.json")
    cases = json.loads(cases_path.read_text())
    if len(cases) != 6 or len({c["id"] for c in cases}) != 6:
        raise ValueError("Exactly the six fixed comparison cases required")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    plan = {
        "scope": __doc__, "script_sha256": sha(Path(__file__)),
        "shared_benchmark_sha256": sha(Path(__file__).with_name("benchmark_serving.py")),
        "cases_sha256": sha(cases_path), "cases": cases, "repeats": REPEATS,
        "measured_requests": 96, "cells": CELLS,
        "order": "Repeat outer, case inner; rotate four cells left by (repeat + case index) modulo 4",
        "warmup": "One retained 32-token request per cell in declared cell order, prospectively excluded",
        "primary": "Fast marked/ordinary seconds per committed token; same one-sided 95% bootstrap upper <= 1.05 screen",
        "secondary": "Reference marked/ordinary and fast/reference per condition; descriptive",
        "randomness": "Independent SDK cryptographic draws, not identical generated paths across cells",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "uncertainty": "Within-prompt paired bootstrap with four repeats; fixed workload only",
        "timed_scope": "Prompt encoding, model calls, filters, marking, sampling, durable journals, reports",
        "excluded_scope": "Model/candidate setup, imports, inspection, HTTP, streaming, native-server baseline",
        "model_sharing": "One verified MLX model/tokenizer shared sequentially; fresh response caches; one bound candidate per execution",
        "memory": "MLX active/peak/cache per request; process RSS is cumulative",
        "failure_rule": "Retain all errors, caps, warmups; no retries, replacement or outlier removal",
        "minimum_free_bytes": MIN_FREE_BYTES, "free_bytes_before": free,
        "idle_host_confirmed_by_operator": True,
        "platform": platform.platform(), "python": platform.python_version(),
    }
    write(public / "plan.json", plan)
    rows, warmups, failure, analysis = [], [], None, None
    try:
        import mlx.core as mx
        from keyprint import Keyprint
        from keyprint.experimental.fast_public import FastPublicCandidate
        key = Keyprint.new_key()
        with os.fdopen(os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(key)
        mx.synchronize()
        started = time.perf_counter()
        reference = Keyprint.from_mlx(args.model, key=key)
        fast = Keyprint(key=key)
        fast._candidate = FastPublicCandidate(fast._candidate)
        fast._backend = reference._backend
        candidates = {"reference": reference, "fast": fast}
        mx.synchronize()
        write(public / "load.json", {
            "seconds": time.perf_counter() - started,
            "identity": {name: c.identity for name, c in candidates.items()},
            "versions": {p: importlib.metadata.version(p) for p in ("keyprint", "mlx", "mlx-lm", "numpy")},
            "scope": "One verified model load plus both candidates; imports excluded; filesystem cache may be warm",
        })
        def attempt(case, repeat, execution, condition, warmup=False):
            name = f"{execution}-{'warmup' if warmup else case['id']}-{repeat}-{condition}"
            row = {"id": name, "case": case["id"], "repeat": repeat, "execution": execution,
                   "condition": condition, "max_tokens": 32 if warmup else case["max_tokens"]}
            mx.synchronize()
            mx.reset_peak_memory()
            row["active_bytes_before"] = mx.get_active_memory()
            started = time.perf_counter()
            try:
                result = candidates[execution].generate(case["prompt"], max_tokens=row["max_tokens"],
                                                       condition=condition, output=args.output / name)
                mx.synchronize()
                row["seconds"] = time.perf_counter() - started
                tokens = result.report["usage"]["completion_tokens"]
                if tokens != len(result.report["payload"]["committed_token_ids"]):
                    raise ValueError("Usage and committed path differ")
                import hashlib
                row.update(text=result.text, text_sha256=hashlib.sha256(result.text.encode()).hexdigest(),
                           completion_tokens=tokens, completion=result.report["payload"]["completion"],
                           report_sha256=sha(args.output / name / "report.json"),
                           journal_sha256=sha(args.output / name / "journal.jsonl"))
            except Exception as exc:
                row.update(seconds=time.perf_counter() - started, error={"type": type(exc).__name__})
            row.update(active_bytes_after=mx.get_active_memory(), peak_bytes=mx.get_peak_memory(),
                       cache_bytes_after=mx.get_cache_memory(), process_cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write(public / (name + ".json"), row)
            print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
            return row
        for execution, condition in CELLS:
            warmups.append(attempt(cases[0], -1, execution, condition, True))
        if any("error" in row for row in warmups):
            raise ValueError("Warmup failed; no acceptance timings collected")
        for repeat in range(REPEATS):
            for index, case in enumerate(cases):
                for execution, condition in cell_order(repeat, index):
                    rows.append(attempt(case, repeat, execution, condition))
        analysis = analyze(rows, [c["id"] for c in cases])
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    summary = {"status": "completed" if failure is None else "incomplete", "failure": failure,
               "attempts": len(rows), "warmups": len(warmups), "analysis": analysis,
               "production_overhead_accepted": False}
    write(public / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())
