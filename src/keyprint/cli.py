"""Small, offline-first command line interface."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys


def comparison() -> dict:
    import numpy as np
    from . import Keyprint

    candidate = Keyprint(key=bytes(range(32)))
    reports = {}
    for condition in ("ordinary", "marked"):
        draws = random.Random(7)
        with candidate.pipeline(condition=condition) as pipeline:
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


def doctor() -> dict:
    from . import verify
    problems, packages = [], {}
    for name in ("keyprint", "numpy", "scipy", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Missing package: {name}")
    try:
        integrity = verify()
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import tokenizers  # noqa: F401
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        integrity = {"status": "fail"}
        problems.append(str(exc))
    return {"status": "fail" if problems else "pass", "packages": packages,
            "python": platform.python_version(), "platform": platform.system(),
            "integrity": integrity, "problems": problems,
            "scope": "Package import and source checks only. No model, API connection, or detection accuracy test."}


def load_key(path: Path) -> bytes:
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise ValueError("key file must be private; run chmod 600 on it")
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("key file must contain 32 raw bytes; use keyprint keygen")
    return key


def main(argv: list[str] | None = None) -> int:
    from . import Keyprint, KeyprintError, __version__, verify
    parser = argparse.ArgumentParser(prog="keyprint", description="Generate and inspect text watermarks locally.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    for name, help_text in (("demo", "Try an offline sampling example"),
                            ("doctor", "Check this installation"),
                            ("verify", "Verify the bundled engine")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--json", action="store_true")
    keygen = commands.add_parser("keygen", help="Create a private key without printing it")
    keygen.add_argument("path", type=Path, nargs="?", default=Path("keyprint.key"))
    generate = commands.add_parser("generate", help="Generate text with a supported local model")
    generate.add_argument("--backend", choices=("mlx", "transformers"), default="mlx")
    generate.add_argument("--model", type=Path, required=True, help="Local model directory")
    generate.add_argument("--key", type=Path, required=True, help="Private key from keyprint keygen")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=64)
    generate.add_argument("--condition", choices=("ordinary", "marked"), default="marked")
    generate.add_argument("--output", type=Path)
    serve = commands.add_parser("serve", help="Serve a local, text-only OpenAI client endpoint")
    serve.add_argument("--backend", choices=("mlx", "transformers"), default="mlx")
    serve.add_argument("--model", type=Path, required=True)
    serve.add_argument("--key", type=Path, required=True)
    serve.add_argument("--api-key", type=Path, required=True, help="A separate private key file; clients use its hex encoding")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--output", type=Path, default=Path("private-keyprint-server"))
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            parser.print_help()
        elif args.command == "keygen":
            descriptor = os.open(args.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(Keyprint.new_key())
                stream.flush()
                os.fsync(stream.fileno())
            print(f"Private key created: {args.path}\nKeep it private and backed up.")
        elif args.command in ("generate", "serve"):
            loader = Keyprint.from_mlx if args.backend == "mlx" else Keyprint.from_transformers
            key = load_key(args.key)
            if args.command == "serve":
                import uvicorn
                from .server import create_app
                token = load_key(args.api_key)
                if token == key:
                    raise ValueError("API and watermark keys must be different")
                if not 1024 <= args.port <= 65535:
                    raise ValueError("port must be between 1024 and 65535")
                app = create_app(lambda: loader(args.model, key=key), api_key=token.hex(), output=args.output)
                print(f"Local preview: http://127.0.0.1:{args.port}/v1 (one user text message; no streaming)")
                uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
                return 0
            candidate = loader(args.model, key=key)
            result = candidate.generate(args.prompt, max_tokens=args.max_tokens,
                                        condition=args.condition, output=args.output)
            print(result.text)
            print(f"\nPrivate report: {result.artifacts / 'report.json'}", file=sys.stderr)
        elif args.command == "demo":
            reports = comparison()
            scope = "Supplied A/B/C/D logits, public demonstration key, seeded draws. No model or detection verdict."
            if args.json:
                print(json.dumps({"scope": scope, "reports": reports}, allow_nan=False))
            else:
                print("Keyprint | offline sampling demo\n")
                for condition, report in reports.items():
                    print(f"{condition.capitalize():8}  {report.get('rendered_carriers', {}).get('visible_text', 'Failed')}")
                print(f"\n{scope}\nNext: keyprint generate --help")
            return int(any(report["kind"] == "error" for report in reports.values()))
        else:
            result = doctor() if args.command == "doctor" else verify()
            if args.json:
                print(json.dumps(result, allow_nan=False))
            else:
                print(f"Keyprint {args.command}: {result['status'].upper()}\n{result['scope']}")
                for problem in result.get("problems", []):
                    print(problem)
            return int(result["status"] != "pass")
        return 0
    except KeyprintError as exc:
        print(f"Keyprint: generation failed. Private report: {exc.artifacts}", file=sys.stderr)
        return 1
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Keyprint: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
