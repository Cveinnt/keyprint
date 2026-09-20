"""Exact top-k development screen; no SDK integration or serving acceptance."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from benchmark_serving import sha
from partition_topk import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    plan = {"scope": __doc__, "width": 151669, "top_k": 100, "repeats": 30,
            "seed": 20260920, "distributions": ["normal", "rounded", "equal"],
            "order": "Alternate reference and candidate first on each repetition",
            "includes": "Threshold partition, tie selection and final sorted selected positions",
            "excludes": "Model inference, other support filtering, sampling, SDK lifecycle and reports",
            "script_sha256": sha(Path(__file__)), "candidate_sha256": sha(Path(__file__).with_name("partition_topk.py"))}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    rng = np.random.default_rng(plan["seed"])
    ids = np.arange(plan["width"])
    results = []
    for kind in plan["distributions"]:
        pairs = []
        for repeat in range(plan["repeats"]):
            scores = rng.normal(size=len(ids)).astype(np.float32).astype(np.float64)
            if kind == "rounded": scores = scores.round(0)
            elif kind == "equal": scores.fill(1.)
            calls = [("reference", lambda: np.lexsort((ids, -scores))[:100]),
                     ("partition", lambda: select(ids, scores, 100))]
            if repeat % 2: calls.reverse()
            outputs, times = {}, {}
            for name, call in calls:
                start = time.perf_counter_ns()
                outputs[name] = call()
                times[name] = time.perf_counter_ns() - start
            if not np.array_equal(outputs["reference"], outputs["partition"]):
                raise ValueError("Selected order changed")
            pairs.append(times)
        results.append({"distribution": kind, "pairs": pairs,
                        "median_partition_over_reference": statistics.median(p["partition"] / p["reference"] for p in pairs)})
    summary = {"status": "pass", "exact_order_comparisons": 90, "results": results,
               "scope": __doc__, "serving_acceptance": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "comparisons": 90,
                      "ratios": {r["distribution"]: r["median_partition_over_reference"] for r in results}}))


if __name__ == "__main__": main()
