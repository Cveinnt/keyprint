"""Exact-arithmetic development screen for vectorized tournament updates.

Not integrated into the SDK. Compare every layer and diagnostic count before
reporting helper timings; these timings exclude model inference and serving.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
from keyprint._engine.legacy._impl.research.byte_trie_numeric import update, MIN_POSITIVE


def update_array(q, bits, diagnostics=None):
    if (not isinstance(q, np.ndarray) or q.dtype != np.float64 or q.ndim != 1 or not q.size
            or not isinstance(bits, np.ndarray) or bits.shape != q.shape or bits.dtype != np.int8
            or not np.isin(bits, (0, 1)).all()):
        raise ValueError("invalid tournament inputs")
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("invalid tournament probability")
    total = math.fsum(q.tolist())
    if total <= 0 or abs(total - 1.) > 1e-12:
        raise ValueError("tournament probabilities must sum to one")
    zero_mass = math.fsum(q[bits == 0].tolist())
    # Separate ufunc operations preserve the oracle's multiply/add rounding.
    # Ignore process-global NumPy error settings; the scalar oracle validates
    # finite results and explicitly floors underflow instead of emitting warnings.
    with np.errstate(all="ignore"):
        weights = q * (zero_mass + bits * total)
        denominator = math.fsum(weights.tolist())
        if denominator <= 0 or not math.isfinite(denominator):
            raise ArithmeticError("invalid tournament normalization")
        output = weights / denominator
    floor = (q > 0) & (output == 0)
    floored = int(np.count_nonzero(floor))
    if floored:
        output[floor] = MIN_POSITIVE
        if diagnostics is not None:
            diagnostics["branch_roundups"] = diagnostics.get("branch_roundups", 0) + floored
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    plan = {"sizes": [1, 10, 100, 1000], "repeats": 40, "layers": 30,
            "seed": 20260920, "scope": __doc__, "order": "Alternate reference-first and candidate-first by repeat"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    rng = random.Random(plan["seed"])
    rows = []
    for n in plan["sizes"]:
        timings = {"reference": [], "candidate": []}
        for rep in range(plan["repeats"]):
            p = [10 ** rng.uniform(-300, 0) for _ in range(n)]
            q = tuple(v / math.fsum(p) for v in p)
            bits = [[rng.randrange(2) for _ in range(n)] for _ in range(30)]
            left, right = q, np.array(q, dtype=np.float64)
            a, b = {}, {}
            for mask in bits:
                left = update(left, mask, a)
                right = update_array(right, np.array(mask, dtype=np.int8), b)
                if np.asarray(left).tobytes() != right.tobytes() or a != b:
                    raise ValueError("Layer probabilities or diagnostic counters differ")
            prepared = np.array(bits, dtype=np.int8)
            for name in (("reference", "candidate") if rep % 2 == 0 else ("candidate", "reference")):
                value = q if name == "reference" else np.array(q, dtype=np.float64)
                started = time.perf_counter_ns()
                if name == "reference":
                    for mask in bits: value = update(value, mask)
                else:
                    for mask in prepared: value = update_array(value, mask)
                timings[name].append(time.perf_counter_ns() - started)
        rows.append({"size": n, "comparisons": 40 * 30,
                     "reference_median_ns": statistics.median(timings["reference"]),
                     "candidate_median_ns": statistics.median(timings["candidate"]),
                     "median_paired_ratio": statistics.median(b/a for a,b in zip(timings["reference"],timings["candidate"]))})
    result = {"status": "pass", "rows": rows, "sdk_integrated": False, "serving_accepted": False}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
