"""Human-readable Keyprint command; legacy JSON CLI remains available."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import sys


def comparison():
    import numpy as np
    from . import PublicCandidate

    candidate = PublicCandidate()
    reports = {}
    for condition in ("ordinary", "marked"):
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
    return reports


def doctor():
    packages = {}
    problems = []
    for name in ("keyprint-research-v3", "numpy", "scipy", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Missing package: {name}")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if check.returncode:
        problems.append((check.stdout + check.stderr).strip())
    # Importing this package has already verified the frozen bundle hashes.
    return {"status": "fail" if problems else "pass", "python": platform.python_version(),
            "platform": platform.system(), "packages": packages, "problems": problems,
            "scope": "Package dependencies and bundle integrity only; no model or API connection tested."}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="keyprint", description="Explore and verify the Keyprint research SDK.")
    parser.add_argument("--version", action="version", version=importlib.metadata.version("keyprint-research-v3"))
    sub = parser.add_subparsers(dest="command")
    for name, help_text in (("demo", "Compare ordinary and marked fixture choices; no model needed"),
                            ("doctor", "Check your local installation"),
                            ("verify", "Verify bundled source integrity")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--json", action="store_true", help="Emit complete machine-readable results")
    generate = sub.add_parser("generate", help="Generate a prose pair using the supported local MLX checkpoint")
    generate.add_argument("--model", type=Path, required=True, help="Local pinned Qwen3-8B-4bit directory")
    generate.add_argument("--prompt", default="Explain why the sky is blue in two short sentences.")
    generate.add_argument("--max-tokens", type=int, default=64)
    generate.add_argument("--output", type=Path, default=Path("private-keyprint-run"))
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "generate":
            from .mlx_generate import main as generate_main
            return generate_main(["--model", str(args.model), "--prompt", args.prompt,
                                  "--max-tokens", str(args.max_tokens), "--output", str(args.output)])
        if args.command == "demo":
            reports = comparison()
            scope = "Supplied A/B/C/D logits, public key and seeded draws. No language model or detection verdict."
            if args.json:
                print(json.dumps({"scope": scope, "reports": reports}, ensure_ascii=False, allow_nan=False))
            else:
                print("Keyprint | offline sampling demo\n")
                for name, report in reports.items():
                    text = (report.get("rendered_carriers") or {}).get("visible_text")
                    print(f"{name.capitalize():8}  {text if text is not None else 'Failed; run with --json for details'}")
                print(f"\n{scope}\nUse keyprint demo --json for full reports.")
            return int(any(report["kind"] == "error" for report in reports.values()))
        if args.command == "doctor":
            result = doctor()
            if args.json:
                print(json.dumps(result))
            else:
                print(f"Keyprint setup: {result['status'].upper()}")
                print(f"Python {result['python']} on {result['platform']}")
                for name, version in result["packages"].items():
                    print(f"  {name}: {version}")
                for problem in result["problems"]:
                    print(problem)
                print(result["scope"])
            return int(result["status"] != "pass")
        from . import __version__
        result = {"status": "pass", "version": __version__,
                  "scope": "Bundled file hashes verified; not an authorship or accuracy check."}
        print(json.dumps(result) if args.json else f"Keyprint {__version__}: bundle integrity verified.\n{result['scope']}")
        return 0
    except ImportError as exc:
        print(f"Keyprint: optional model dependency unavailable ({exc.name}). Install this release's [mlx] extra on Apple Silicon macOS.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Keyprint: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
