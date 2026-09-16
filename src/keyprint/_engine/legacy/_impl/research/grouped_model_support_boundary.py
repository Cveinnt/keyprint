"""Lossless shape projection only when unmapped logits are already -infinity.

Research host boundary, not a mask or integration with the actual model. Apply
after the ordinary host's pre-existing filtering, before the grouped transform.
Finite unmapped logits are errors, even if softmax would underflow them to zero.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "release/tokenizer/qwen3-8b-4bit/config.json"
CONFIG_SHA = "e5485285fd7e289e76e9cffa112f6dc2e3426519082f7db9b69041589f81a218"
TOKENIZER_PATH = ROOT / "release/tokenizer/qwen3-8b-4bit/tokenizer.json"
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def declared_sizes():
    raw, tok = CONFIG_PATH.read_bytes(), TOKENIZER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA or hashlib.sha256(tok).hexdigest() != TOKENIZER_SHA:
        raise ValueError("model/tokenizer configuration hash mismatch")
    config, tokenizer = json.loads(raw), json.loads(tok)
    ids = set(tokenizer["model"]["vocab"].values()) | {t["id"] for t in tokenizer["added_tokens"]}
    if ids != set(range(max(ids) + 1)):
        raise ValueError("non-dense tokenizer mapping")
    model_size, mapped_size = config["vocab_size"], len(ids)
    if model_size < mapped_size:
        raise ValueError("tokenizer exceeds declared model vocabulary")
    return model_size, mapped_size


def project_supported_logits(logits: np.ndarray, *, model_size: int, mapped_size: int) -> np.ndarray:
    if type(model_size) is not int or type(mapped_size) is not int or not 0 < mapped_size <= model_size:
        raise ValueError("invalid declared sizes")
    if not isinstance(logits, np.ndarray) or logits.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("boundary requires NumPy float32/float64 logits")
    if logits.shape != (1, model_size):
        raise ValueError("unexpected model-logit shape")
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("invalid model logits")
    if not np.isneginf(logits[:, mapped_size:]).all():
        raise ValueError("unmapped slots retain finite logits; projection would change support")
    head = logits[:, :mapped_size]
    if not np.isfinite(head).any():
        raise ValueError("mapped vocabulary has no finite support")
    return np.frombuffer(head.tobytes(), dtype=logits.dtype).reshape(1, mapped_size)
