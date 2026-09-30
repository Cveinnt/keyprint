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
        modules = ["fastapi", "uvicorn", *backend_modules(backend)]
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
        elif backend == 'llama-cpp':
            with path.open('rb') as stream:
                if stream.read(4) != b'GGUF':
                    raise ValueError('--model must be a GGUF file')
            result['model_check'] = 'GGUF header present; binding, weights and inference not verified'
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
        raise ValueError('--execution is an MLX option; other backends use their own supported execution')


def execution_argument(parser) -> None:
    parser.add_argument('--execution', choices=('reference', 'experimental-fast', 'experimental-native'),
                        default='reference', help='MLX execution; default: reference. Native requires the reviewed keyprint-native wheel.')


def model_loader(backend: str, execution: str, keyprint):
    validate_execution(backend, execution)
    if backend == 'mlx':
        return lambda path, **settings: keyprint.from_mlx(path, execution=execution, **settings)
    if backend == 'llama-cpp':
        return keyprint.from_llama_cpp
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


def pinned_model(backend: str):
    if backend == "mlx":
        from .backends.mlx import MODEL_ID, REVISION, ASSETS
        return MODEL_ID, REVISION, list(ASSETS), "about 4.62 GB"
    if backend == "transformers":
        return ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac",
                ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json"], "about 270 MB")
    if backend == "llama-cpp":
        from .backends.llama_cpp import MODEL_ID, REVISION, MODEL_FILE
        return MODEL_ID, REVISION, [MODEL_FILE], "about 145 MB"
    raise ValueError("backend must be mlx, transformers or llama-cpp")


def backend_modules(backend):
    return {"mlx": ("mlx.core", "mlx_lm"), "transformers": ("torch", "transformers"),
            "llama-cpp": ("llama_cpp",)}[backend]


def model_asset_path(path: Path, backend: str) -> Path:
    if backend == "llama-cpp":
        from .backends.llama_cpp import MODEL_FILE
        return path / MODEL_FILE
    return path


def validate_model_path(path: Path, backend: str) -> None:
    if backend == 'llama-cpp':
        if not path.is_file():
            raise ValueError('--model must be an existing local GGUF file')
    elif not path.is_dir():
        raise ValueError('--model must be an existing local directory')


def model_cache_root() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))


def download_model(backend: str) -> Path:
    """Explicit public pinned-asset download; never install or execute model code."""
    if backend == "mlx" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        raise ValueError("MLX requires Apple Silicon macOS; choose --backend transformers")
    model_id, revision, files, size = pinned_model(backend)
    modules = ("mlx_lm",) if backend == "mlx" else backend_modules(backend)
    try:
        for name in modules:
            importlib.import_module(name)
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(f"Install backend dependencies first: python -m pip install '.[{backend},server]'") from exc
    print(f"Downloading {model_id} ({size} of model files on first use)\n"
          f"Pinned revision: {revision}\nCache: {model_cache_root()}\n"
          "Existing cached files are reused. Model files are checked when loaded.", flush=True)
    try:
        path = snapshot_download(repo_id=model_id, revision=revision, cache_dir=model_cache_root(),
                                 allow_patterns=files, token=False)
    except Exception as exc:
        raise RuntimeError(f"Pinned model download failed: {exc}\n"
                           "Cached files are retained. Retry --download explicitly when ready, or use --model PATH.") from exc
    return model_asset_path(Path(path), backend)


def cached_model(backend: str) -> Path:
    """Resolve only documented, pinned assets; never download implicitly."""
    model_id, revision, files, _ = pinned_model(backend)
    path = model_cache_root() / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    asset = model_asset_path(path, backend)
    if not (asset.is_file() if backend == "llama-cpp" else (path / "config.json").is_file()):
        command = shlex.join(["hf", "download", model_id, "--revision", revision,
                              "--include", *files])
        raise ValueError(f"No pinned {backend} model cached at {path}.\n"
                         f"From this checkout, install the backend: pip install '.[{backend}]'\n"
                         f"Then download the pinned model explicitly:\n  {command}\n"
                         "For the interactive demo, keyprint playground --download fetches these same pinned files.\n"
                         "Rerun your command afterward, or pass --model PATH for existing local assets.\n"
                         "No download was started. Cached files are verified by the selected backend when loaded.")
    return asset


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
            command.add_argument('--backend', choices=('mlx', 'transformers', 'llama-cpp'))
            command.add_argument('--model', type=Path)
            execution_argument(command)
    keygen = commands.add_parser("keygen", help="Create a private key without printing it")
    keygen.add_argument("path", type=Path, nargs="?", default=Path("keyprint.key"))
    generate = commands.add_parser("generate", help="Generate text with a supported local model")
    generate.add_argument("--backend", choices=("mlx", "transformers", "llama-cpp"), default=default_backend(),
                          help="Default: MLX on Apple Silicon macOS; Transformers elsewhere")
    generate.add_argument("--model", type=Path, help="Local model directory or GGUF file; otherwise use a documented pinned cache")
    generate.add_argument("--key", type=Path, required=True, help="Private key from keyprint keygen")
    generate.add_argument("--prompt", type=prompt_text, required=True)
    generate.add_argument("--max-tokens", type=token_limit, default=64)
    generate.add_argument("--condition", choices=("ordinary", "marked"), default="marked")
    generate.add_argument("--output", type=Path)
    generate.add_argument("--json-schema", type=Path, help="JSON Schema file; MLX or Transformers with [structured]")
    execution_argument(generate)
    playground = commands.add_parser("playground", help="Open a real-model generation and editing playground")
    playground.add_argument("--backend", choices=("mlx", "transformers", "llama-cpp"),
                            default=default_backend(), help="Default: MLX on Apple Silicon macOS; Transformers elsewhere")
    playground.add_argument("--model", type=Path, help="Local model directory or GGUF file; otherwise use a documented pinned cache")
    playground.add_argument("--download", action="store_true", help="Explicitly fetch pinned public model files into the Hugging Face cache before starting")
    playground.add_argument("--key", type=Path, help="Optional private key; otherwise create a fresh session key")
    playground.add_argument("--port", type=int, default=8766)
    playground.add_argument("--output", type=Path, help="Private run directory; otherwise create a new temporary directory")
    execution_argument(playground)
    serve = commands.add_parser("serve", help="Serve local text endpoints for OpenAI, Anthropic and Ollama clients")
    serve.add_argument("--backend", choices=("mlx", "transformers", "llama-cpp"), default=default_backend(),
                       help="Default: MLX on Apple Silicon macOS; Transformers elsewhere")
    serve.add_argument("--model", type=Path, help="Local model directory or GGUF file; otherwise use a documented pinned cache")
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
            if args.download and args.model is not None:
                raise ValueError("Use either --download for the pinned model or --model for existing local files")
            if not 1024 <= args.port <= 65535:
                raise ValueError("port must be between 1024 and 65535")
            try:
                import uvicorn
                from .playground import create_playground
            except ImportError as exc:
                raise ImportError("The playground needs the [server] extra. From this checkout: pip install '.[server]'") from exc
            key = load_key(args.key) if args.key else None
            if args.download and args.execution == "experimental-native":
                from .experimental.native_mlx import native_backend
                native_backend()
            path = (download_model(args.backend) if args.download else
                    args.model if args.model is not None else cached_model(args.backend))
            validate_model_path(path, args.backend)
            directory = args.output or Path(tempfile.mkdtemp(prefix="keyprint-playground-"))
            key = Keyprint.new_key() if key is None else key
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
            validate_model_path(path, args.backend)
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
            try:
                result = candidate.generate(args.prompt, max_tokens=args.max_tokens,
                                            condition=args.condition, output=args.output,
                                            **({"json_schema": schema} if args.json_schema is not None else {}))
            except BaseException:
                try:
                    candidate.close()
                except Exception as cleanup_error:
                    # Retain the generation failure and its artifact path even
                    # if releasing backend-owned resources also fails.
                    print(f"Keyprint: backend cleanup also failed: {cleanup_error}", file=sys.stderr)
                raise
            else:
                candidate.close()
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
