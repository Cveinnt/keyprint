"""Real model smoke check, retaining every attempt and private key locally.

This is integration evidence, not quality or detection calibration. Run against
an installed wheel, not PYTHONPATH=src. Never commit the output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from keyprint import Keyprint, KeyprintError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    key = Keyprint.new_key()
    fd = os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(key)
    loader = Keyprint.from_mlx if args.backend == "mlx" else Keyprint.from_transformers
    candidate = loader(args.model, key=key)
    results = []
    report = {"backend": args.backend, "version": importlib.metadata.version("keyprint"),
              "identity": candidate.identity, "runs": results,
              "scope": "Real model smoke test only; no quality, cost, or detector acceptance."}
    prompts = ["Explain why the sky is blue in two short sentences.",
               "Write a short email asking to move a meeting from Tuesday to Wednesday.",
               "En español, explica en dos frases por qué dormimos."]
    for index, prompt in enumerate(prompts):
        for condition in ("ordinary", "marked"):
            start = time.monotonic()
            try:
                result = candidate.generate(prompt, condition=condition, max_tokens=128,
                                            output=args.output / f"{index}-{condition}")
                try:
                    score = candidate.score(result.text)
                except (KeyprintError, ValueError) as exc:
                    score = {"kind": "unavailable", "error_type": type(exc).__name__}
                entry = {"prompt": prompt, "condition": condition, "text": result.text,
                         "text_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                         "report": result.report, "literal_diagnostic": score}
            except KeyprintError as exc:
                entry = {"prompt": prompt, "condition": condition, "report": exc.report}
            entry["elapsed_seconds"] = time.monotonic() - start
            results.append(entry)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            print(index, condition, entry["report"]["kind"], round(entry["elapsed_seconds"], 2), flush=True)
    return int(any(run["report"]["kind"] == "error" for run in results))


if __name__ == "__main__":
    raise SystemExit(main())
