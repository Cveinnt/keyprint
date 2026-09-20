"""Pinned MLX backend; loads only explicitly supplied local assets."""
from pathlib import Path
import hashlib
import platform
from typing import Any
from ..errors import InputLimitError

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



class MLXModel:
    def __init__(self, model: Any, tokenizer: Any):
        self.model, self.tokenizer = model, tokenizer

    @classmethod
    def load(cls, path: Path):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("MLX requires Apple Silicon macOS")
        verify_assets(path)
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError("Install Keyprint with the [mlx] extra") from exc
        model, tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
        return cls(model, tokenizer)

    def encode_prompt(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
            raise ValueError("prompt must contain 1 to 16000 characters")
        ids = self.tokenizer.apply_chat_template([{ "role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not ids or any(type(i) is not int or not 0 <= i < 151669 for i in ids):
            raise ValueError("prompt is outside the supported model binding")
        if len(ids) > 8192:
            raise InputLimitError(input_tokens=len(ids), limit=8192)
        return ids
