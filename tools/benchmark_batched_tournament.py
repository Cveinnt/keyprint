"""Development screen: batch tournament rounds while preserving scalar arithmetic.

Validate immutable bit tables once. Keep math.fsum, operation order, per-round
normalization checks and explicit subnormal flooring. Not an SDK integration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
from keyprint._engine.legacy._impl.research.byte_trie_numeric import update, MIN_POSITIVE


def update_layers(q, bits, diagnostics=None, *, observe=None):
    if (not isinstance(q, np.ndarray) or q.dtype != np.float64 or q.ndim != 1 or not q.size
            or not isinstance(bits, np.ndarray) or bits.dtype != np.int8 or bits.ndim != 2
            or bits.shape[1] != q.size or not bits.shape[0] or ((bits < 0) | (bits > 1)).any()):
        raise ValueError("invalid tournament inputs")
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("invalid tournament probability")
    r = q.copy()
    zero_masks = bits == 0
    floats = bits.astype(np.float64)
    with np.errstate(all="ignore"):
        for zero_mask, layer in zip(zero_masks, floats):
            total = math.fsum(r.tolist())
            if total <= 0 or abs(total - 1.) > 1e-12:
                raise ValueError("tournament probabilities must sum to one")
            zero_mass = math.fsum(r[zero_mask].tolist())
            weights = r * (zero_mass + layer * total)
            denominator = math.fsum(weights.tolist())
            if denominator <= 0 or not math.isfinite(denominator):
                raise ArithmeticError("invalid tournament normalization")
            output = weights / denominator
            floor = (r > 0) & (output == 0)
            floored = int(np.count_nonzero(floor))
            if floored:
                output[floor] = MIN_POSITIVE
                if diagnostics is not None:
                    diagnostics["branch_roundups"] = diagnostics.get("branch_roundups", 0) + floored
            r = output
            if observe is not None: observe(r, diagnostics)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    plan = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sizes": [1, 10, 32, 63, 64, 100, 1000], "repeats": 40, "layers": 30,
            "seed": 20260921, "scope": __doc__, "order": "Alternate reference/candidate first by repeat"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    rng = random.Random(plan["seed"])
    rows = []
    for size in plan["sizes"]:
        timings = {"reference": [], "candidate": []}
        for repeat in range(plan["repeats"]):
            p = [10 ** rng.uniform(-300, 0) for _ in range(size)]
            q = tuple(v / math.fsum(p) for v in p)
            bits = [[rng.randrange(2) for _ in range(size)] for _ in range(plan["layers"])]
            reference, counters, expected = q, {}, []
            for layer in bits:
                reference = update(reference, layer, counters)
                expected.append((np.asarray(reference).tobytes(), counters.copy()))
            observed = []
            update_layers(np.asarray(q), np.asarray(bits, dtype=np.int8), {},
                          observe=lambda result, counts: observed.append((result.tobytes(), counts.copy())))
            if observed != expected: raise ValueError("Layer probabilities or counters differ")
            for name in (("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")):
                started = time.perf_counter_ns()
                if name == "reference":
                    result = q
                    for layer in bits: result = update(result, layer)
                else:
                    result = update_layers(np.asarray(q), np.asarray(bits, dtype=np.int8))
                timings[name].append(time.perf_counter_ns() - started)
        rows.append({"size": size, "layer_comparisons": plan["repeats"] * plan["layers"],
                     "reference_median_ns": statistics.median(timings["reference"]),
                     "candidate_median_ns": statistics.median(timings["candidate"]),
                     "median_paired_ratio": statistics.median(b/a for a,b in zip(timings["reference"],timings["candidate"]))})
    summary = {"status": "pass", "rows": rows, "sdk_integrated": False, "serving_accepted": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
