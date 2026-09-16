"""Small typed API over the explicitly namespaced reference implementation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import secrets
import tempfile

from .integrity import verify


class KeyprintError(RuntimeError):
    """A failed operation retains its report; it is never a negative verdict."""
    def __init__(self, report: dict[str, Any], artifacts: Path | None = None):
        self.report = report
        self.artifacts = artifacts
        super().__init__("Keyprint operation failed; inspect .report for the phase and consumed work")


@dataclass(frozen=True)
class Generation:
    text: str
    report: dict[str, Any]
    artifacts: Path


class Keyprint:
    """One owner's watermarking context with an explicit private key.

    The current reference backend has a pinned Qwen tokenizer. Unsupported
    model bindings fail instead of borrowing its evidence or detector settings.
    """
    def __init__(self, *, key: bytes, temperature: float = 0.7, top_k: int = 100):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("key must be exactly 32 bytes; use Keyprint.new_key()")
        verify()
        from ._engine.research.keyprint_v3_public_api_rc2 import PublicCandidate
        self._candidate = PublicCandidate(temperature=temperature, top_k=top_k)
        self._key = key
        self._backend: Any = None

    @staticmethod
    def new_key() -> bytes:
        return secrets.token_bytes(32)

    @classmethod
    def from_mlx(cls, model: str | Path, *, key: bytes, **settings: Any) -> Keyprint:
        instance = cls(key=key, **settings)
        from .backends.mlx import MLXModel
        instance._backend = MLXModel.load(Path(model))
        return instance

    @classmethod
    def from_transformers(cls, model: str | Path, *, key: bytes,
                          temperature: float = .7, top_k: int = 100) -> Keyprint:
        """Load a local text-only model with an experimental ByteLevel profile."""
        instance = cls(key=key, temperature=temperature, top_k=top_k)
        from .backends.transformers import TransformersModel
        instance._backend = TransformersModel.load(Path(model), temperature=temperature, top_k=top_k)
        return instance

    @property
    def identity(self) -> dict[str, Any]:
        if hasattr(self._backend, "identity"):
            return self._backend.identity
        return self._candidate.core_identity

    def score(self, text: str) -> dict[str, Any]:
        """Return an uncalibrated matching-key diagnostic, never authorship."""
        if hasattr(self._backend, "score"):
            return self._backend.score(text, self._key)
        report = self._candidate.score_literal(text, self._key)
        if report["kind"] == "error":
            raise KeyprintError(report)
        return report

    def pipeline(self, *, condition: str = "marked", **settings: Any) -> Any:
        """Advanced supplied-logit API; see model binding and commit contract."""
        if hasattr(self._backend, "identity"):
            raise ValueError("pipeline() is only available for the reference binding; use generate() for Transformers")
        return self._candidate.pipeline(self._key, condition=condition, **settings)

    def generate(self, prompt: str, *, max_tokens: int = 64,
                 condition: str = "marked", output: str | Path | None = None) -> Generation:
        if self._backend is None:
            raise ValueError("load a supported backend first, for example Keyprint.from_mlx(...)")
        if condition not in ("ordinary", "marked"):
            raise ValueError("condition must be ordinary or marked")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024")
        if hasattr(self._backend, "generate"):
            return self._backend.generate(prompt, key=self._key, max_tokens=max_tokens,
                                          condition=condition, output=output)
        ids = self._backend.encode_prompt(prompt)
        return self._run(self._backend.model, ids, max_tokens=max_tokens,
                         condition=condition, output=output)

    def _run(self, model: Callable[..., Any], ids: list[int], *, max_tokens: int,
             condition: str, output: str | Path | None,
             backend: Any = None, cache_factory: Any = None) -> Generation:
        from ._engine.research.keyprint_candidate_v3_caller import DurableJournal
        directory = Path(tempfile.mkdtemp(prefix="keyprint-")) if output is None else Path(output)
        if output is not None:
            directory.mkdir(mode=0o700)
        reservations: dict[str, int] = {}

        def reserve(action: str, metadata: dict[str, Any]) -> None:
            reservations[action] = reservations.get(action, 0) + 1

        with DurableJournal(directory / "journal.jsonl") as journal:
            report = self._candidate.run_response(
                model, ids, key=self._key, condition=condition,
                random_bits=secrets.randbits, journal=journal, reserve=reserve,
                max_tokens=max_tokens, max_model_calls=max_tokens + 8,
                allow_thinking=False, allow_tools=False,
                backend=backend, cache_factory=cache_factory,
            )
        report["package_scope"] = "Namespaced reference port; no new model-family or scientific acceptance."
        with (directory / "report.json").open("x", encoding="utf-8") as stream:
            json.dump({"report": report, "reservations": reservations}, stream, ensure_ascii=False, allow_nan=False)
        if report["kind"] == "error":
            raise KeyprintError(report, directory)
        return Generation(report["rendered_carriers"]["visible_text"], report, directory)
