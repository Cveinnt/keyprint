"""Small, offline-first command line interface."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import secrets
import shlex
import sys
import tempfile


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


def doctor(*, playground=False, backend=None, model=None, execution="reference") -> dict:
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
    result = {"status": "fail" if problems else "pass", "packages": packages,
            "python": platform.python_version(), "platform": platform.system(),
            "integrity": integrity, "problems": problems,
            "scope": "Package import and source checks only. No model, API connection, or detection accuracy test."}
    if not playground:
        return result
    backend = backend or default_backend()
    result.update(workflow="playground", backend=backend, execution=execution,
                  scope="Playground dependency and local-file preflight. No model loading, inference, hosted API calls, or detection acceptance.")
    try:
        validate_execution(backend, execution)
        if backend == "mlx" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
            raise RuntimeError("MLX requires Apple Silicon macOS; choose --backend transformers")
        modules = ["fastapi", "uvicorn", *(('mlx.core', 'mlx_lm') if backend == 'mlx' else ('torch', 'transformers'))]
        for module in modules:
            try:
                importlib.import_module(module)
            except (ImportError, RuntimeError, OSError) as exc:
                problems.append(f"Cannot import {module}: {exc}")
        if execution == "experimental-native":
            from .experimental.native_mlx import native_backend
            result['native_accelerator'] = native_backend().identity['package_version']
        path = Path(model) if model is not None else cached_model(backend)
        result['model'] = str(path.resolve())
        if backend == 'mlx':
            from .backends.mlx import verify_assets
            verify_assets(path)
            result['model_check'] = 'Pinned asset hashes verified; model not loaded'
        else:
            if not path.is_dir():
                raise ValueError('--model must be an existing local directory')
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
                json.loads((path / name).read_text())
            if not any(path.glob('*.safetensors')):
                raise ValueError('Local safetensors weights are missing')
            result['model_check'] = 'Required files present and metadata parses; binding, weights and inference not verified'
        result['next_command'] = shlex.join(['keyprint', 'playground', '--backend', backend,
            '--model', str(path.resolve()), '--execution', execution])
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        problems.append(str(exc))
    if problems:
        result['install_hint'] = f"From the reviewed source checkout: python -m pip install '.[{backend},server]'"
    result['status'] = 'fail' if problems else 'pass'
    return result


def load_key(path: Path) -> bytes:
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise ValueError("key file must be private; run chmod 600 on it")
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("key file must contain 32 raw bytes; use keyprint keygen")
    return key


def default_backend() -> str:
    return "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "transformers"


def validate_execution(backend: str, execution: str) -> None:
    if execution != 'reference' and backend != 'mlx':
        raise ValueError('--execution is an MLX option; Transformers uses its own supported execution')


def execution_argument(parser) -> None:
    parser.add_argument('--execution', choices=('reference', 'experimental-fast', 'experimental-native'),
                        default='reference', help='MLX execution; default: reference. Native requires the reviewed keyprint-native wheel.')


def model_loader(backend: str, execution: str, keyprint):
    validate_execution(backend, execution)
    if backend == 'mlx':
        return lambda path, **settings: keyprint.from_mlx(path, execution=execution, **settings)
    return keyprint.from_transformers


def token_limit(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-tokens must be an integer from 1 to 1024") from exc
    if not 1 <= count <= 1024:
        raise argparse.ArgumentTypeError("max-tokens must be an integer from 1 to 1024")
    return count


def prompt_text(value: str) -> str:
    if not value.strip() or len(value) > 16000:
        raise argparse.ArgumentTypeError("prompt must contain 1 to 16000 characters and not be blank")
    return value


def cached_model(backend: str) -> Path:
    """Resolve only documented, pinned assets; never download implicitly."""
    models = {
        "mlx": ("mlx-community--Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
        "transformers": ("HuggingFaceTB--SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    }
    root = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    name, revision = models[backend]
    path = root / ("models--" + name) / "snapshots" / revision
    if not (path / "config.json").is_file():
        model_id = name.replace("--", "/", 1)
        command = shlex.join(["hf", "download", model_id, "--revision", revision,
                              "--include", "*.json", "*.safetensors", "*.jinja"])
        raise ValueError(f"No pinned {backend} model cached at {path}.\n"
                         f"From this checkout, install the backend: pip install '.[{backend}]'\n"
                         f"Then download the pinned model explicitly:\n  {command}\n"
                         "Rerun your command afterward, or pass --model PATH for existing local assets.\n"
                         "No download was started. Cached files are verified by the selected backend when loaded.")
    return path


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
        if name == 'doctor':
            command.add_argument('--playground', action='store_true', help='Check real-demo dependencies and local model files without loading weights')
            command.add_argument('--backend', choices=('mlx', 'transformers'))
            command.add_argument('--model', type=Path)
            execution_argument(command)
    keygen = commands.add_parser("keygen", help="Create a private key without printing it")
    keygen.add_argument("path", type=Path, nargs="?", default=Path("keyprint.key"))
    generate = commands.add_parser("generate", help="Generate text with a supported local model")
    generate.add_argument("--backend", choices=("mlx", "transformers"), default=default_backend(),
                          help="Default: MLX on Apple Silicon macOS; Transformers elsewhere")
    generate.add_argument("--model", type=Path, help="Local model directory; otherwise use a documented pinned cache")
    generate.add_argument("--key", type=Path, required=True, help="Private key from keyprint keygen")
    generate.add_argument("--prompt", type=prompt_text, required=True)
    generate.add_argument("--max-tokens", type=token_limit, default=64)
    generate.add_argument("--condition", choices=("ordinary", "marked"), default="marked")
    generate.add_argument("--output", type=Path)
    generate.add_argument("--json-schema", type=Path, help="JSON Schema file; MLX or Transformers with [structured]")
    execution_argument(generate)
    playground = commands.add_parser("playground", help="Open a real-model generation and editing playground")
    playground.add_argument("--backend", choices=("mlx", "transformers"),
                            default=default_backend(), help="Default: MLX on Apple Silicon macOS; Transformers elsewhere")
    playground.add_argument("--model", type=Path, help="Local model directory; otherwise use a documented pinned cache")
    playground.add_argument("--key", type=Path, help="Optional private key; otherwise create a fresh session key")
    playground.add_argument("--port", type=int, default=8766)
    playground.add_argument("--output", type=Path, help="Private run directory; otherwise create a new temporary directory")
    execution_argument(playground)
    serve = commands.add_parser("serve", help="Serve local text endpoints for OpenAI and Anthropic clients")
    serve.add_argument("--backend", choices=("mlx", "transformers"), default=default_backend(),
                       help="Default: MLX on Apple Silicon macOS; Transformers elsewhere")
    serve.add_argument("--model", type=Path, help="Local model directory; otherwise use a documented pinned cache")
    serve.add_argument("--key", type=Path, required=True)
    serve.add_argument("--api-key", type=Path, required=True, help="A separate private key file; clients use its hex encoding")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--output", type=Path, default=Path("private-keyprint-server"))
    execution_argument(serve)
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            parser.print_help()
        elif args.command == "playground":
            loader = model_loader(args.backend, args.execution, Keyprint)
            try:
                import uvicorn
                from .playground import create_playground
            except ImportError as exc:
                raise ImportError("The playground needs the [server] extra. From this checkout: pip install '.[server]'") from exc
            path = args.model if args.model is not None else cached_model(args.backend)
            if not path.is_dir():
                raise ValueError("--model must be an existing local directory")
            if not 1024 <= args.port <= 65535:
                raise ValueError("port must be between 1024 and 65535")
            directory = args.output or Path(tempfile.mkdtemp(prefix="keyprint-playground-"))
            key = load_key(args.key) if args.key else Keyprint.new_key()
            token = secrets.token_urlsafe(32)
            app = create_playground(lambda: loader(path, key=key), token=token, output=directory, port=args.port)
            if args.key is None:
                descriptor = os.open(directory / "session.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(key)
            print(f"Keyprint playground\nModel: {path}\nPrivate artifacts: {directory}\n"
                  f"Open now: http://127.0.0.1:{args.port}/#session={token}\n"
                  "The prefilled example runs once on page load. No hosted API calls.\n"
                  "Keep the session URL private. Ctrl+C stops the server.", flush=True)
            uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
            return 0
        elif args.command == "keygen":
            descriptor = os.open(args.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(Keyprint.new_key())
                stream.flush()
                os.fsync(stream.fileno())
            print(f"Private key created: {args.path}\nKeep it private and backed up.")
        elif args.command in ("generate", "serve"):
            loader = model_loader(args.backend, args.execution, Keyprint)
            key = load_key(args.key)
            path = args.model if args.model is not None else cached_model(args.backend)
            if not path.is_dir():
                raise ValueError("--model must be an existing local directory")
            if args.command == "serve":
                import uvicorn
                from .server import create_app
                token = load_key(args.api_key)
                if token == key:
                    raise ValueError("API and watermark keys must be different")
                if not 1024 <= args.port <= 65535:
                    raise ValueError("port must be between 1024 and 65535")
                app = create_app(lambda: loader(path, key=key), api_key=token.hex(), output=args.output)
                print(f"Local preview: http://127.0.0.1:{args.port}/v1 (one user text message; no streaming)")
                uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
                return 0
            schema = json.loads(args.json_schema.read_text()) if args.json_schema is not None else None
            candidate = loader(path, key=key)
            result = candidate.generate(args.prompt, max_tokens=args.max_tokens,
                                        condition=args.condition, output=args.output,
                                        **({"json_schema": schema} if args.json_schema is not None else {}))
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
                print(f"\n{scope}\nReal-model interactive demo: keyprint playground --help")
            return int(any(report["kind"] == "error" for report in reports.values()))
        else:
            if args.command == 'doctor':
                if not args.playground and (args.backend is not None or args.model is not None or args.execution != 'reference'):
                    raise ValueError('Use doctor --playground to check backend, model or execution settings')
                if args.playground and not args.json:
                    print('Checking playground installation and local model files...', flush=True)
                result = doctor(playground=args.playground, backend=args.backend, model=args.model, execution=args.execution)
            else:
                result = verify()
            if args.json:
                print(json.dumps(result, allow_nan=False))
            else:
                print(f"Keyprint {args.command}: {result['status'].upper()}\n{result['scope']}")
                for problem in result.get("problems", []):
                    print(problem)
                if result.get('install_hint'):
                    print(result['install_hint'])
                if result['status'] == 'pass' and result.get('next_command'):
                    print(f"Next: {result['next_command']}")
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
