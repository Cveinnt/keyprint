"""Compare portable sampling arithmetic, not model throughput or service latency.

Run from an installed checkout: python tools/benchmark_sampling.py --output result.json
Both implementations receive identical synthetic post-filter logits, marked
weights and random-bit seeds. Every measured output must match the dense law.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from keyprint.sampling import identity, sparse_sample, sparse_softmax
from keyprint._engine.research.keyprint_exact_categorical_v2 import sample_float_weights, supported_softmax
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession


def benchmark(width, repeats):
    rng = np.random.default_rng(20260917)
    logits = np.full(width, -np.inf, dtype=np.float64)
    logits[rng.choice(width, 100, replace=False)] = rng.uniform(-40, 0, 100)
    profile = Profile([str(i).encode() for i in range(width)],
                      tokenizer_identity="synthetic-benchmark", config=Config(max_steps=1))
    session = SparseTokenSourceSession(profile, bytes(range(32)), condition="marked")
    try:
        weights = session.prepare(sparse_softmax(logits)).probabilities
        def dense(seed):
            return (np.array(supported_softmax(tuple(map(float, logits)))),
                    sample_float_weights(tuple(map(float, weights)), random.Random(seed).getrandbits))
        def sparse(seed):
            return sparse_softmax(logits), sparse_sample(weights, random.Random(seed).getrandbits)
        implementations = {"dense": dense, "sparse": sparse}
        for function in implementations.values():
            function(0)
        times = {name: [] for name in implementations}
        for seed in range(repeats):
            results = {}
            order = ("dense", "sparse") if seed % 2 == 0 else ("sparse", "dense")
            for name in order:
                started = time.perf_counter()
                results[name] = implementations[name](seed)
                times[name].append(time.perf_counter() - started)
            assert results["dense"][0].tobytes() == results["sparse"][0].tobytes()
            assert asdict(results["dense"][1]) == asdict(results["sparse"][1])
        medians = {name: statistics.median(values) for name, values in times.items()}
        return {"vocabulary": width, "positive_candidates": 100, "repeats": repeats,
                "seconds": times, "median_seconds": medians,
                "median_speedup": medians["dense"] / medians["sparse"], "exact_parity": True}
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 100:
        parser.error("repeats must be between 2 and 100")
    report = {"scope": "Synthetic softmax plus exact categorical draw only; excludes model, filtering, transform and journaling",
              "sampling_execution": identity(), "python": platform.python_version(),
              "platform": platform.platform(), "numpy": importlib.metadata.version("numpy"),
              "results": [benchmark(width, args.repeats) for width in (49152, 151669)]}
    if args.output:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
