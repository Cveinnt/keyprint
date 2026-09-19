"""Fixed public-corpus null screen; no model calls, detector verdict or release approval.

Two independent keys per split. The unit is one response's maximum across keys,
not two independent observations. Thresholds freeze before held-out scoring.
Only public/ is exportable; source response text is not copied into artifacts.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import unicodedata

import numpy as np
from compare_score_baselines import score

SOURCE = "https://huggingface.co/datasets/databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
SALT = "keyprint-null-corpus-v1"
WEIGHTS = {"uniform": np.ones(30), "linear_10_to_1": np.linspace(10, 1, 30)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def select_rows(rows, count):
    """Group transitive shared instructions, contexts or responses before splitting."""
    eligible = [(i, r) for i, r in enumerate(rows) if 100 <= len(r["response"].split()) <= 400]
    parent = list(range(len(eligible)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, (_, row) in enumerate(eligible):
        for field in ("instruction", "context", "response"):
            value = normalized(row[field])
            if not value:
                continue
            fingerprint = (field, sha(value.encode()))
            if fingerprint in seen:
                parent[root(i)] = root(seen[fingerprint])
            seen[fingerprint] = i
    groups = {}
    for i, pair in enumerate(eligible):
        groups.setdefault(root(i), []).append(pair)
    representatives = [min(group, key=lambda p: p[0]) for group in groups.values()]
    representatives.sort(key=lambda p: sha((SALT + normalized(p[1]["response"])).encode()))
    if len(representatives) < count:
        raise ValueError("Insufficient deduplicated responses")
    selected = [{"source_index": i, "category": row["category"], "text": row["response"],
                 "text_sha256": sha(row["response"].encode()),
                 "words": len(row["response"].split())} for i, row in representatives[:count]]
    return selected, {"source_rows": len(rows), "length_eligible": len(eligible),
                      "independent_exact_groups": len(groups),
                      "limitation": "Exact normalized shared fields only; near duplicates/authors/topics may remain dependent"}


def threshold(values, target):
    """Predeclared conservative empirical order statistic; strict exceedances only."""
    if not 0 < target < 1 or not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Finite scores and a target between zero and one required")
    rank = math.ceil((len(values) + 1) * (1 - target))
    if rank > len(values):
        raise ValueError("Insufficient calibration size for target")
    return sorted(values)[rank - 1]


def iid_binomial_upper(hits, total, alpha=.025):
    """One-sided exact upper bound under IID Bernoulli assumptions, not corpus proof."""
    if not 0 <= hits <= total or total < 1 or not 0 < alpha < 1:
        raise ValueError("Valid binomial counts required")
    if hits == total:
        return 1.0
    if hits == 0:
        return -math.expm1(math.log(alpha) / total)
    low, high = 0.0, 1.0
    for _ in range(60):
        p = (low + high) / 2
        logs = [math.lgamma(total + 1) - math.lgamma(i + 1) - math.lgamma(total - i + 1)
                + i * math.log(p) + (total - i) * math.log1p(-p) for i in range(hits + 1)]
        largest = max(logs)
        cdf = math.exp(largest) * math.fsum(math.exp(v - largest) for v in logs)
        if cdf > alpha:
            low = p
        else:
            high = p
    return high


def summarize(rows, thresholds, expected_count):
    panels = {}
    for name, cutoff in thresholds.items():
        usable = [r for r in rows if not r.get("error") and name in r.get("maximum", {})]
        hits = sum(r["maximum"][name] > cutoff for r in usable)
        categories = {}
        for category in sorted({r["category"] for r in rows}):
            group = [r for r in usable if r["category"] == category]
            categories[category] = {"available": len(group), "above": sum(r["maximum"][name] > cutoff for r in group)}
        panels[name] = {"threshold": cutoff, "planned": expected_count, "available": len(usable),
                        "unavailable": expected_count - len(usable), "above": hits,
                        "iid_only_upper_97_5_percent": iid_binomial_upper(hits, len(usable))
                        if len(usable) == expected_count else None, "categories": categories}
    return panels


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_split(binding, selected, keys, output):
    rows = []
    with output.open("x") as stream:
        for item in selected:
            row = {k: v for k, v in item.items() if k != "text"}
            try:
                scores = []
                for key in keys:
                    events = binding.replay_text(item["text"], key).events
                    bits = np.asarray([e.bits for e in events if e.eligible])
                    scores.append({name: score(bits, weights) for name, weights in WEIGHTS.items()})
                row.update(scores=scores, eligible_events=len(bits),
                           maximum={name: max(s[name] for s in scores) for name in WEIGHTS})
            except Exception as exc:
                row["error"] = type(exc).__name__
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if len(rows) % 50 == 0:
                print(json.dumps({"split": output.stem, "attempted": len(rows),
                                  "errors": sum(bool(r.get('error')) for r in rows)}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if sha(raw) != SOURCE_SHA256:
        raise ValueError("Pinned public dataset checksum differs")
    selected, selection = select_rows([json.loads(line) for line in raw.splitlines()], 1000)
    baseline = json.loads(args.baseline.read_text())
    old_thresholds = {name: baseline["panels"][name]["threshold"] for name in WEIGHTS}
    if not all(math.isfinite(v) for v in old_thresholds.values()):
        raise ValueError("Finite original thresholds required")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    keys = {split: [secrets.token_bytes(32) for _ in range(2)] for split in ("calibration", "heldout")}
    for split, values in keys.items():
        for index, key in enumerate(values):
            with os.fdopen(os.open(args.output / f"{split}-{index}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                stream.write(key)
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding = runtime_binding(max_steps=2048)
    plan = {"scope": "English public human-response null screen; not generated-output quality or deployment qualification",
            "source": SOURCE, "revision": REVISION, "source_sha256": SOURCE_SHA256,
            "attribution": "Copyright (2023) Databricks, Inc.; Wikipedia editors and contributors; CC BY-SA 3.0",
            "license": "https://creativecommons.org/licenses/by-sa/3.0/",
            "script_sha256": sha(Path(__file__).read_bytes()),
            "score_source_sha256": sha(Path(__file__).with_name('compare_score_baselines.py').read_bytes()),
            "baseline_sha256": sha(args.baseline.read_bytes()), "binding": binding.profile.identity_receipt(),
            "selection": selection, "split_salt": SALT, "min_words": 100, "max_words": 400,
            "calibration_count": 500, "heldout_count": 500, "target": .01,
            "unit": "maximum across two independent keys for each response; document count is denominator",
            "threshold_rule": "order statistic ceil((n+1)*(1-target)); strictly greater-than; no tuning",
            "uncertainty": "97.5% one-sided binomial upper bounds per scorer assume IID documents; corpus dependencies and cross-key shifts prevent deployment guarantee",
            "key_commitments": {s: [sha(k) for k in ks] for s, ks in keys.items()},
            "weights": {n: w.tolist() for n, w in WEIGHTS.items()}, "original_thresholds": old_thresholds,
            "records": [{**{k: v for k, v in item.items() if k != 'text'},
                         "split": "calibration" if i < 500 else "heldout"} for i, item in enumerate(selected)]}
    write(public / "plan.json", plan)
    calibration = run_split(binding, selected[:500], keys["calibration"], public / "calibration.jsonl")
    if any(r.get("error") for r in calibration):
        write(public / "summary.json", {"status": "incomplete", "reason": "Calibration scoring unavailable; no threshold or held-out run"})
        return 1
    frozen = {n: threshold([r["maximum"][n] for r in calibration], .01) for n in WEIGHTS}
    write(public / "thresholds.json", frozen)
    heldout = run_split(binding, selected[500:], keys["heldout"], public / "heldout.jsonl")
    complete = not any(r.get("error") for r in heldout)
    write(public / "summary.json", {"status": "completed" if complete else "incomplete",
          "new_thresholds": summarize(heldout, frozen, 500),
          "original_thresholds": summarize(heldout, old_thresholds, 500),
          "deployment_calibrated": False, "power_established": False,
          "scope": plan["scope"], "uncertainty": plan["uncertainty"]})
    print(json.dumps({"status": "completed" if complete else "incomplete", "thresholds": frozen}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
