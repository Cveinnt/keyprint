"""Compare eight supplied-logit steps. This fixture does not generate prose."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
from keyprint_v3 import PublicCandidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fixture-comparison.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new --output path")
    reports = {}
    candidate = PublicCandidate()
    for condition in ("ordinary", "marked"):
        # Repeated public key and seeded draws are for this illustration only.
        # Actual experiments require private keys and independent unbiased bits.
        draws = random.Random(7)
        with candidate.pipeline(bytes(range(32)), condition=condition) as pipeline:
            for _ in range(8):
                head = np.full((1, 151936), -np.inf, dtype=np.float32)
                head[0, 32:36] = 0
                report = pipeline.step(head, draws.getrandbits)
                if report["kind"] == "error":
                    break
            else:
                report = pipeline.finish()
        reports[condition] = report
    artifact = {
        "scope": "Eight A/B/C/D supplied-logit steps; public key and seeded fixture draws. No model, semantic-quality test or detection efficacy claim.",
        "reports": reports,
    }
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    print(artifact["scope"])
    for condition, report in reports.items():
        text = report.get("rendered_carriers", {}).get("visible_text")
        print(f"{condition:8}: {text!r} ({report['kind']})")
    print(f"Full reports and interpretation: {args.output}")
    print("No author, AI-provider or Claude-detection verdict follows from these outputs.")
    return int(any(r["kind"] == "error" for r in reports.values()))


if __name__ == "__main__":
    raise SystemExit(main())
