"""Paired real-inference timing of marked versus ordinary within the MLX SDK.

Run on an otherwise idle host. This measures incremental marking cost inside
the same SDK, not total SDK overhead against a native production server. Keep
all texts, failures and caps. No quality or A18 release acceptance is inferred.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import time

import numpy as np

REPEATS = 4
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260920
OVERHEAD_LIMIT = .05
MIN_FREE_BYTES = 2 * 1024**3


def require_storage(output):
    """Reject a full destination before creating receipts or loading weights."""
    ancestor = output.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < MIN_FREE_BYTES:
        raise ValueError("Serving benchmark requires at least 2 GiB free on the output filesystem")
    return free


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, case_ids):
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Nonempty distinct case IDs required")
    expected = {(case,repeat,condition) for case in case_ids for repeat in range(REPEATS)
                for condition in ("ordinary","marked")}
    available = {(r["case"],r["repeat"],r["condition"]):r for r in rows}
    if len(rows)!=len(expected) or set(available)!=expected or any("error" in r for r in rows):
        raise ValueError("Every planned pair is required; no failed or replaced attempts")
    token_logs, request_logs = [], []
    for case in case_ids:
        token_group, request_group = [], []
        for repeat in range(REPEATS):
            ordinary,marked = (available[case,repeat,c] for c in ("ordinary","marked"))
            for row in (ordinary,marked):
                if (type(row["completion_tokens"]) is not int or row["completion_tokens"]<1
                        or type(row["seconds"]) not in (float,int)
                        or not math.isfinite(row["seconds"]) or row["seconds"]<=0):
                    raise ValueError("Positive measured time and actual token count required")
            ratio = marked["seconds"]/ordinary["seconds"]
            request_group.append(math.log(ratio))
            token_group.append(math.log(ratio*ordinary["completion_tokens"]/marked["completion_tokens"]))
        token_logs.append(token_group)
        request_logs.append(request_group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0,REPEATS,size=(BOOTSTRAPS,len(case_ids),REPEATS))
    def interval(groups):
        values = np.asarray(groups)
        sampled = values[np.arange(len(case_ids))[None,:,None],indices].mean(axis=(1,2))
        return {"geometric_mean_ratio":float(np.exp(values.mean())),
                "one_sided_95_upper_ratio":float(np.exp(np.quantile(sampled,.95)))}
    per_token,per_request = interval(token_logs),interval(request_logs)
    return {"pairs":len(expected)//2,"seconds_per_committed_token":per_token,
            "request_latency":per_request,
            "incremental_5pct_timing_screen_pass":per_token["one_sided_95_upper_ratio"]<=1+OVERHEAD_LIMIT,
            "uncertainty":"Paired bootstrap within each fixed prompt; four pairs per prompt. Approximate finite-workload timing uncertainty, not workload or deployment coverage.",
            "native_server_overhead_measured":False,"production_overhead_accepted":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--idle-host-confirmed",action="store_true",
                        help="Operator has stopped other inference/benchmark workloads; does not prove OS isolation")
    args = parser.parse_args()
    if not args.idle_host_confirmed:
        raise ValueError("Do not benchmark concurrently with other inference workloads")
    free_bytes_before = require_storage(args.output)
    cases_path = Path(__file__).with_name("inference_cases.json")
    cases = json.loads(cases_path.read_text())
    if len(cases)!=6 or len({c["id"] for c in cases})!=6:
        raise ValueError("Exactly the six fixed public comparison cases required")
    args.output.mkdir(mode=0o700)
    public = args.output/"public"
    public.mkdir()
    def write(path,value):
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    plan = {"scope":"Local incremental marking cost within the same SDK; no native-server, output-quality or production acceptance",
            "script_sha256":sha(Path(__file__)),"cases_sha256":sha(cases_path),"cases":cases,
            "repeats":REPEATS,"pairs":24,"warmup":"One 32-token ordinary and one 32-token marked request, excluded prospectively from timing ratios but retained",
            "order":"Repeat outer loop, six cases in file order; ordinary first on even repeat+case index, marked first on odd",
            "randomness":"Independent SDK cryptographic sampling; not paired identical text or random draws",
            "primary":"Geometric mean marked/ordinary seconds per committed token; bootstrap complete pairs within each fixed prompt",
            "secondary":"Request latency ratio includes differences in generated length; descriptive",
            "incremental_timing_screen":"One-sided 95% bootstrap upper ratio <= 1.05; does not approve total SDK overhead or A18",
            "bootstrap_replicates":BOOTSTRAPS,"bootstrap_seed":BOOTSTRAP_SEED,
            "timed_scope":"Prompt encoding, model calls, filtering, watermarking, sampling, durable journals and report serialization",
            "excluded_scope":"Model loading (separately timed), text inspection, HTTP transport, streaming and external native-server baseline",
            "memory":"MLX active/peak/cache bytes per request; OS process peak RSS is cumulative and cannot be compared as per-request peak",
            "startup":"Fresh process model load with explicit asset verification; package import time excluded and filesystem cache may already be warm",
            "failure_rule":"Retain all errors, warmups, EOS and capped responses; no retries, outlier trimming or excluding short responses",
            "minimum_free_bytes":MIN_FREE_BYTES,"free_bytes_before":free_bytes_before,
            "idle_host_confirmed_by_operator":True,"platform":platform.platform(),"python":platform.python_version()}
    write(public/"plan.json",plan)
    rows,warmups,failure = [],[],None
    load = None
    try:
        import mlx.core as mx
        from keyprint import Keyprint
        key = Keyprint.new_key()
        with os.fdopen(os.open(args.output/"owner.key",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as stream:
            stream.write(key)
        mx.synchronize()
        mx.reset_peak_memory()
        started = time.perf_counter()
        candidate = Keyprint.from_mlx(args.model,key=key)
        mx.synchronize()
        load = {"seconds":time.perf_counter()-started,"active_bytes":mx.get_active_memory(),
                "peak_bytes":mx.get_peak_memory(),"cache_bytes":mx.get_cache_memory(),
                "identity":candidate.identity,
                "versions":{p:importlib.metadata.version(p) for p in ("keyprint","mlx","mlx-lm","numpy")}}
        write(public/"load.json",load)
        def attempt(case,repeat,condition,*,warmup=False):
            name = f"{'warmup' if warmup else case['id']}-{repeat}-{condition}"
            row = {"id":name,"case":case["id"],"repeat":repeat,"condition":condition,
                   "max_tokens":32 if warmup else case["max_tokens"]}
            mx.synchronize()
            mx.reset_peak_memory()
            row["active_bytes_before"] = mx.get_active_memory()
            started = time.perf_counter()
            try:
                result = candidate.generate(case["prompt"],max_tokens=row["max_tokens"],condition=condition,output=args.output/name)
                mx.synchronize()
                row["seconds"] = time.perf_counter()-started
                payload = result.report.get("payload",result.report)
                tokens = result.report["usage"]["completion_tokens"]
                if tokens!=len(payload["committed_token_ids"]):
                    raise ValueError("Usage differs from committed generation path")
                row.update(text=result.text,text_sha256=hashlib.sha256(result.text.encode()).hexdigest(),
                           completion_tokens=tokens,completion=payload["completion"],
                           report_sha256=sha(args.output/name/"report.json"),
                           journal_sha256=sha(args.output/name/"journal.jsonl"))
            except Exception as exc:
                row["seconds"] = time.perf_counter()-started
                row["error"] = {"type":type(exc).__name__}
            row.update(active_bytes_after=mx.get_active_memory(),peak_bytes=mx.get_peak_memory(),
                       cache_bytes_after=mx.get_cache_memory(),process_cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write(public/(name+".json"),row)
            print(json.dumps({k:v for k,v in row.items() if k!="text"}),flush=True)
            return row
        for condition in ("ordinary","marked"):
            warmups.append(attempt(cases[0],-1,condition,warmup=True))
        if any("error" in r for r in warmups):
            raise ValueError("Warmup failure; do not collect acceptance timings")
        for repeat in range(REPEATS):
            for index,case in enumerate(cases):
                order = ("ordinary","marked") if (repeat+index)%2==0 else ("marked","ordinary")
                for condition in order:
                    rows.append(attempt(case,repeat,condition))
        analysis = summarize(rows,[c["id"] for c in cases])
    except Exception as exc:
        failure = {"type":type(exc).__name__,"message":str(exc)}
        analysis = None
    summary = {"status":"completed" if failure is None else "incomplete","attempts":len(rows),
               "warmups":len(warmups),"failure":failure,"analysis":analysis,
               "scope":plan["scope"],"production_overhead_accepted":False}
    write(public/"summary.json",summary)
    print(json.dumps(summary),flush=True)
    return 0 if failure is None else 1


if __name__=="__main__":
    raise SystemExit(main())
