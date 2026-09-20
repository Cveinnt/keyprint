"""Development-only SHA-256 HMAC context execution, compared with stdlib HMAC.

Uses the standard inner/outer pad construction for the SDK's fixed 32-byte keys.
This is a helper experiment, not an SDK integration or security certification.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import statistics
import time

from benchmark_prf_context import reused_context_bits
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile, pack


class SHAContext:
    def __init__(self, key, prefix, layers):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("exactly 32 key bytes required")
        padded = key + bytes(32)
        self.outer = hashlib.sha256(bytes(v ^ 0x5c for v in padded))
        base = hashlib.sha256(bytes(v ^ 0x36 for v in padded))
        base.update(prefix)
        self.templates = []
        for layer in range(layers):
            inner = base.copy()
            inner.update(layer.to_bytes(4, "big"))
            self.templates.append(inner)

    def digest(self, layer, suffix):
        inner = self.templates[layer].copy()
        inner.update(suffix)
        outer = self.outer.copy()
        outer.update(inner.digest())
        return outer.digest()


def sha_context_bits(profile, key, context, labels):
    context_bytes = pack(context)
    prefix = ((4).to_bytes(4, "big") + len(profile._domain).to_bytes(8, "big") + profile._domain
              + len(context_bytes).to_bytes(8, "big") + context_bytes + (4).to_bytes(8, "big"))
    engine = SHAContext(key, prefix, profile.config.layers)
    result = {}
    for label in labels:
        if not label: raise ValueError("empty classes have no watermark bits")
        suffix = len(label).to_bytes(8, "big") + label
        result[label] = tuple(engine.digest(i, suffix)[0] & 1 for i in range(profile.config.layers))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    plan = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "label_counts": [1, 10, 100], "context_lengths": [0, 1, 4], "layers": 30, "repeats": 60,
            "reference": "Existing reusable hmac.HMAC context helper; verify against Profile.bits first",
            "scope": __doc__, "order": "Alternate reference/candidate order by repeat"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    profile = Profile([f"label-{i}".encode() for i in range(100)], tokenizer_identity="sha-context",
                      config=Config(layers=30, max_steps=1024))
    rows = []
    for count in plan["label_counts"]:
        labels = profile.classes[:count]
        for length in plan["context_lengths"]:
            context = tuple(f"prior-{i}".encode() for i in range(length))
            timings = {"reference": [], "candidate": []}
            for repeat in range(plan["repeats"]):
                key = bytes((i + repeat) % 256 for i in range(32))
                expected = {label: profile.bits(key, context, label) for label in labels}
                for name in (("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")):
                    fn = reused_context_bits if name == "reference" else sha_context_bits
                    started = time.perf_counter_ns()
                    actual = fn(profile, key, context, labels)
                    timings[name].append(time.perf_counter_ns() - started)
                    if actual != expected: raise ValueError("PRF bits differ")
            rows.append({"labels": count, "context_length": length, "comparisons": 60,
                         "median_paired_ratio": statistics.median(b/a for a,b in zip(timings["reference"],timings["candidate"]))})
    summary = {"status": "pass", "comparisons": 540, "rows": rows, "sdk_integrated": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
