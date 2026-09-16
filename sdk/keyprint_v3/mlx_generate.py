"""Generate an ordinary/marked prose pair with the explicitly supported MLX model.

No automatic downloads, hosted API calls, retries, or detector verdicts.
The output directory contains private keys and random draws. Do not publish it.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import sys
import time

MODEL_ID = "mlx-community/Qwen3-8B-4bit"
REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
ASSETS = {
    "config.json": "e5485285fd7e289e76e9cffa112f6dc2e3426519082f7db9b69041589f81a218",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "model.safetensors": "f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8",
}


def verify_assets(model_path):
    if not model_path.is_dir():
        raise ValueError("--model must be an existing local model directory")
    if (model_path / "generation_config.json").exists():
        raise ValueError("unexpected generation_config.json; use the pinned revision")
    if sorted(p.name for p in model_path.glob("*.safetensors")) != ["model.safetensors"]:
        raise ValueError("expected exactly the pinned model.safetensors")
    for name, expected in ASSETS.items():
        with (model_path / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"model asset mismatch: {name}; use the pinned revision")


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local pinned Qwen3-8B-4bit directory")
    parser.add_argument("--prompt", default="Explain why the sky is blue in two short sentences.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("private-keyprint-run"))
    args = parser.parse_args(argv)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this example requires Apple Silicon macOS; use compare.py for offline fixtures")
    if not 1 <= args.max_tokens <= 256:
        parser.error("--max-tokens must be between 1 and 256 per condition")
    if not args.prompt.strip() or len(args.prompt) > 16000:
        parser.error("--prompt must contain 1 to 16000 characters")
    if args.output.exists():
        parser.error("--output already exists; select a fresh directory, never overwrite a run")
    verify_assets(args.model)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    from mlx_lm import load
    from keyprint_v3 import DurableJournal, PublicCandidate

    print("Loading verified local model...", flush=True)
    model, tokenizer = load(str(args.model), tokenizer_config={"trust_remote_code": False})
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    if not prompt_ids or len(prompt_ids) > 8192 or any(type(i) is not int or not 0 <= i < 151669 for i in prompt_ids):
        parser.error("prompt tokenization is outside the SDK's bound prompt contract")
    candidate = PublicCandidate()
    args.output.mkdir(mode=0o700)
    key = secrets.token_bytes(32)
    descriptor = os.open(args.output / "key.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(key)
        stream.flush()
        os.fsync(stream.fileno())
    save(args.output / "run.json", {
        "model": MODEL_ID, "revision": REVISION, "asset_sha256": ASSETS,
        "prompt": args.prompt, "max_tokens_per_condition": args.max_tokens,
        "python": sys.version, "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("keyprint-research-v3", "mlx", "mlx-lm", "numpy", "scipy", "tokenizers", "transformers")},
        "randomness": "independent secrets.randbits draws in each condition; same fresh private key",
        "interpretation": "One local integration smoke test, not a quality comparison or authorship verdict.",
    })
    failed = False
    for condition in ("ordinary", "marked"):
        print(f"Generating {condition} (up to {args.max_tokens} tokens)...", flush=True)
        started = time.monotonic()
        # The SDK enforces model-call, draw and token caps. Reserve records
        # external resource requests; this local example has no paid API budget.
        reservations = {}

        def reserve(action, metadata):
            reservations[action] = reservations.get(action, 0) + 1

        with DurableJournal(args.output / f"{condition}.jsonl") as journal:
            report = candidate.run_response(
                model, prompt_ids, key=key, condition=condition,
                random_bits=secrets.randbits, journal=journal, reserve=reserve,
                max_tokens=args.max_tokens, allow_thinking=False, allow_tools=False,
                max_model_calls=args.max_tokens + 8,
            )
        save(args.output / f"{condition}.json", {
            "report": report, "reservations": reservations,
            "elapsed_seconds": time.monotonic() - started,
            "timing_is_serving_benchmark": False,
        })
        if report["kind"] == "error":
            failed = True
            print(f"{condition}: failed; preserved report and journal in {args.output}", file=sys.stderr)
            break
        print(f"{condition}: {report['rendered_carriers']['visible_text']}", flush=True)
    print(f"Private artifacts: {args.output}. Do not publish the key or journals.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
