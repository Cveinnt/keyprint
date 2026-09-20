"""Frozen-candidate confirmation on eight deterministically selected Dolly prompts.

New to the timing workload, not a claim that the dataset was unseen in research.
Retain original tasks, context, generated text, caps and failures. No production,
semantic-quality or native-server acceptance follows from this timing screen.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

from benchmark_serving import (BOOTSTRAPS, BOOTSTRAP_SEED, MIN_FREE_BYTES, REPEATS,
                               require_storage, sha, summarize)

CORPUS_SHA = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
CORPUS_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SELECTION = "One record per category with lowest SHA256 of keyprint-serving-confirmation-v1:category:zero-based-index; nonempty instruction and formatted prompt <=8000 characters; no output-dependent selection"


def select_cases(path):
    if sha(path)!=CORPUS_SHA:
        raise ValueError("Pinned Dolly source hash differs")
    categories = {}
    for index,line in enumerate(path.read_text().splitlines()):
        record = json.loads(line)
        instruction, context, category = record["instruction"],record["context"],record["category"]
        prompt = instruction + ("\n\nContext:\n"+context if context else "")
        if not instruction.strip() or len(prompt)>8000: continue
        rank = hashlib.sha256(f"keyprint-serving-confirmation-v1:{category}:{index}".encode()).hexdigest()
        case = {"id":f"dolly-{category}-{index}","category":category,"source_index":index,
                "source_record_sha256":hashlib.sha256(line.encode()).hexdigest(),"selection_rank":rank,
                "prompt":prompt,"max_tokens":192,
                "review":"Review the original task and context against both outputs. No semantic approval is inferred from timing."}
        if category not in categories or rank<categories[category][0]: categories[category]=(rank,case)
    if len(categories)!=8: raise ValueError("Exactly eight Dolly categories required")
    return [categories[c][1] for c in sorted(categories)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--idle-host-confirmed",action="store_true",
                        help="Operator has stopped other inference/benchmark workloads; does not prove OS isolation")
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--expected-runtime",required=True)
    parser.add_argument("--execution", choices=["experimental-fast", "experimental-native"],
                        default="experimental-fast")
    args = parser.parse_args()
    if not args.idle_host_confirmed:
        raise ValueError("Do not benchmark concurrently with other inference workloads")
    free_bytes_before = require_storage(args.output)
    if len(args.expected_runtime)!=64 or any(c not in "0123456789abcdef" for c in args.expected_runtime):
        raise ValueError("Exact frozen runtime SHA-256 required")
    cases = select_cases(args.corpus)
    args.output.mkdir(mode=0o700)
    public = args.output/"public"
    public.mkdir()
    def write(path,value):
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    plan = {"scope":"Local incremental marking cost within the same SDK; no native-server, output-quality or production acceptance",
            "script_sha256":sha(Path(__file__)),"shared_benchmark_sha256":sha(Path(__file__).with_name("benchmark_serving.py")),
            "corpus_sha256":CORPUS_SHA,"corpus_revision":CORPUS_REVISION,"corpus_license":"CC-BY-SA-3.0",
            "attribution":"Databricks, databricks-dolly-15k. Prompts format the original instruction and context; responses are newly generated.",
            "source_url":"https://huggingface.co/datasets/databricks/databricks-dolly-15k",
            "selection":SELECTION,"cases":cases,"expected_runtime_sha256":args.expected_runtime,
            "execution":args.execution,
            "repeats":REPEATS,"pairs":32,"warmup":"One 32-token ordinary and marked request for each of eight cases; all sixteen excluded prospectively but retained",
            "order":"Repeat outer loop, eight categories in lexical order; ordinary first on even repeat+case index, marked first on odd",
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
        candidate = Keyprint.from_mlx(args.model,key=key,execution=args.execution)
        if candidate.identity["runtime_profile_sha256"]!=args.expected_runtime:
            raise ValueError("Loaded runtime differs from the frozen confirmation candidate")
        mx.synchronize()
        load = {"seconds":time.perf_counter()-started,"active_bytes":mx.get_active_memory(),
                "peak_bytes":mx.get_peak_memory(),"cache_bytes":mx.get_cache_memory(),
                "identity":candidate.identity,
                "versions":{p:importlib.metadata.version(p) for p in ("keyprint","mlx","mlx-lm","numpy")}}
        write(public/"load.json",load)
        def attempt(case,repeat,condition,*,warmup=False):
            name = f"{'warmup-' if warmup else ''}{case['id']}-{repeat}-{condition}"
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
        for case in cases:
            for condition in ("ordinary","marked"):
                warmups.append(attempt(case,-1,condition,warmup=True))
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


if __name__ == "__main__":
    raise SystemExit(main())
