"""Development-only HMAC context reuse; frozen SDK remains unchanged.

Public fixture keys, no cross-request cache, exact comparison against the
existing bit table. Synthetic timings cannot close the serving-overhead gate.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import statistics
import time

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile, pack
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import bit_table, REFERENCE_BITS


def reused_context_bits(profile, key, context, labels):
    if getattr(profile.bits, "__func__", None) is not REFERENCE_BITS:
        return {label: profile.bits(key, context, label) for label in labels}
    context_bytes = pack(context)
    prefix = ((4).to_bytes(4, "big") + len(profile._domain).to_bytes(8, "big") + profile._domain
              + len(context_bytes).to_bytes(8, "big") + context_bytes + (4).to_bytes(8, "big"))
    base = hmac.new(key, prefix, "sha256")
    templates = []
    for layer in range(profile.config.layers):
        template = base.copy()
        template.update(layer.to_bytes(4, "big"))
        templates.append(template)
    result = {}
    for label in labels:
        suffix = len(label).to_bytes(8, "big") + label
        bits = []
        for template in templates:
            item = template.copy()
            item.update(suffix)
            bits.append(item.digest()[0] & 1)
        result[label] = tuple(bits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    declaration = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "label_counts": [1, 10, 100], "context_lengths": [0, 1, 4], "layers": 30,
                   "repeats": 60, "order": "Alternating reference/candidate order by repeat",
                   "keys": "Public synthetic fixture bytes; key changes with repeat",
                   "acceptance": "Exact bits required; timing is diagnostic only, no SDK promotion or serving acceptance"}
    (args.output / "plan.json").write_text(json.dumps(declaration, indent=2))
    profile = Profile([f"label-{i}".encode() for i in range(100)], tokenizer_identity="prf-fixture",
                      config=Config(layers=30, max_steps=1024))
    results = []
    for count in declaration["label_counts"]:
        labels = list(profile.classes[:count])
        for length in declaration["context_lengths"]:
            context = tuple(f"prior-{i}".encode() for i in range(length))
            timings = {"reference": [], "candidate": []}
            for repeat in range(declaration["repeats"]):
                key = bytes((i + repeat) % 256 for i in range(32))
                order = ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
                values = {}
                for name in order:
                    function = bit_table if name == "reference" else reused_context_bits
                    started = time.perf_counter()
                    values[name] = function(profile, key, context, labels)
                    timings[name].append(time.perf_counter() - started)
                if values["reference"] != values["candidate"]:
                    raise ValueError("Candidate changed reference bits")
            medians = {k: statistics.median(v) for k, v in timings.items()}
            results.append({"labels": count, "context_length": length, "exact_parity": True,
                            "seconds": timings, "median_seconds": medians,
                            "candidate_over_reference": medians["candidate"] / medians["reference"]})
    summary = {"comparisons": 540, "results": results, "sdk_promoted": False,
               "production_overhead_accepted": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({**summary, "results": [{k: v for k, v in row.items() if k != "seconds"} for row in results]}))


if __name__ == "__main__":
    main()
