"""Small typed API over the explicitly namespaced reference implementation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import secrets
import tempfile
from threading import Event

from .integrity import verify
from .rewrite import Rewrite
from .inspection import Inspection
from .cancellation import check_cancellation, _CancellationRequested


class KeyprintError(RuntimeError):
    """A failed operation retains its report; it is never a negative verdict."""
    def __init__(self, report: dict[str, Any], artifacts: Path | None = None):
        self.report = report
        self.artifacts = artifacts
        super().__init__("Keyprint operation failed; inspect .report for the phase and consumed work")


class KeyprintCancelled(KeyprintError):
    """Generation stopped at a cooperative boundary; receipts remain available."""
    def __init__(self, report: dict[str, Any], artifacts: Path | None = None):
        super().__init__(report, artifacts)
        self.args = ("Keyprint generation cancelled; inspect .report for consumed work and .artifacts for receipts",)


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
        self._closed = False

    @staticmethod
    def new_key() -> bytes:
        return secrets.token_bytes(32)

    @classmethod
    def from_mlx(cls, model: str | Path, *, key: bytes, execution: str = "reference", **settings: Any) -> Keyprint:
        """Load pinned MLX; experimental-fast is an explicit execution opt-in."""
        if execution not in ("reference", "experimental-fast", "experimental-native"):
            raise ValueError("execution must be reference, experimental-fast or experimental-native")
        instance = cls(key=key, **settings)
        if execution == "experimental-native":
            from .experimental.native_mlx import NativePublicCandidate
            instance._candidate = NativePublicCandidate(instance._candidate)
        elif execution == "experimental-fast":
            from .experimental.fast_public import FastPublicCandidate
            instance._candidate = FastPublicCandidate(instance._candidate)
        else:
            from .backends.mlx_bounded import BoundedReferencePublicCandidate
            instance._candidate = BoundedReferencePublicCandidate(instance._candidate)
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

    @classmethod
    def from_llama_cpp(cls, model: str | Path, *, key: bytes, temperature: float = .7,
                      top_k: int = 100, context_size: int = 2048, threads: int = 2) -> Keyprint:
        """Load a local CPU GGUF with an experimental byte-BPE binding.

        Single request at a time. Use as a context manager to release native
        resources. Qualification is model-specific; this is not an Ollama API.
        """
        instance = cls(key=key, temperature=temperature, top_k=top_k)
        from .backends.llama_cpp import LlamaCppModel
        instance._backend = LlamaCppModel.load(Path(model), temperature=temperature,
            top_k=top_k, context_size=context_size, threads=threads)
        return instance

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("this Keyprint instance is closed")

    def close(self) -> None:
        """Close backend-owned resources where supported; disable further use."""
        if not self._closed:
            closer = getattr(self._backend, "close", None)
            if closer is not None:
                closer()
            self._closed = True

    def __enter__(self) -> Keyprint:
        self._ensure_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def score(self, text: str) -> dict[str, Any]:
        """Return an uncalibrated matching-key diagnostic, never authorship."""
        return self._score(text, self._key)

    def inspect(self, text: str, *, key: bytes | None = None) -> Inspection:
        """Inspect visible-text counts, optionally using another key as a control.

        This does not generate text or call a hosted API. No detector threshold
        or confidence is inferred from the observed fraction of one bits.
        """
        chosen = self._key if key is None else key
        if type(chosen) is not bytes or len(chosen) != 32:
            raise ValueError("key must be exactly 32 bytes")
        return Inspection.from_report(self._score(text, chosen))

    def _score(self, text: str, key: bytes) -> dict[str, Any]:
        self._ensure_open()
        if not isinstance(text, str) or len(text) > 16000:
            raise ValueError("text must be a string of at most 16000 characters")
        if hasattr(self._backend, "score"):
            return self._backend.score(text, key)
        report = self._candidate.score_literal(text, key)
        if report["kind"] == "error":
            raise KeyprintError(report)
        return report

    def pipeline(self, *, condition: str = "marked", **settings: Any) -> Any:
        """Advanced supplied-logit API; see model binding and commit contract."""
        self._ensure_open()
        if hasattr(self._backend, "identity"):
            raise ValueError("pipeline() is only available for the reference binding; use generate() for other backends")
        return self._candidate.pipeline(self._key, condition=condition, **settings)

    def generate(self, prompt: str, *, max_tokens: int = 64,
                 condition: str = "marked", output: str | Path | None = None,
                 cancel_event: Event | None = None,
                 json_schema: dict[str, Any] | None = None) -> Generation:
        """Generate once; an optional Event requests cooperative cancellation.

        Set the event from another thread. A model call already in progress may
        finish; cancellation is checked before the next call or sample. A
        stopped attempt raises KeyprintCancelled with retained receipts. Use a
        fresh Event for each attempt; do not clear or reuse a requested event.
        json_schema enables constrained JSON on MLX and Transformers with
        the [structured] extra. Token-capped results remain incomplete.
        """
        self._ensure_open()
        if self._backend is None:
            raise ValueError("load a supported backend first, for example Keyprint.from_mlx(...)")
        if condition not in ("ordinary", "marked"):
            raise ValueError("condition must be ordinary or marked")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024")
        if cancel_event is not None and not isinstance(cancel_event, Event):
            raise TypeError("cancel_event must be a threading.Event or None")
        constraint = None
        if json_schema is not None:
            from .backends.transformers import TransformersModel
            from .backends.mlx import MLXModel
            if isinstance(self._backend, MLXModel):
                from .structured import JsonConstraint
                constraint = JsonConstraint.for_mlx(json_schema, self._backend.tokenizer)
            elif not isinstance(self._backend, TransformersModel):
                raise ValueError("json_schema requires the MLX or Transformers backend")
        if hasattr(self._backend, "generate"):
            return self._backend.generate(prompt, key=self._key, max_tokens=max_tokens,
                                          condition=condition, output=output, cancel_event=cancel_event,
                                          **({"json_schema": json_schema} if json_schema is not None else {}))
        ids = self._backend.encode_prompt(prompt)
        return self._run(self._backend.model, ids, max_tokens=max_tokens,
                         condition=condition, output=output, cancel_event=cancel_event,
                         constraint=constraint)

    def rewrite(self, text: str, *, max_tokens: int = 256,
                output: str | Path | None = None, condition: str = "marked",
                preserve: list[str] | tuple[str, ...] = (),
                cancel_event: Event | None = None) -> Rewrite:
        """Unavailable: raises RewriteUnavailableError before inference.

        Language and meaning preservation are not validated. The signature is
        retained so existing callers receive an actionable error.
        """
        from .rewrite import rewrite
        return rewrite(self, text, max_tokens=max_tokens, output=output, condition=condition,
                       preserve=preserve, cancel_event=cancel_event)

    def rewrite_openai(self, response: Any, **settings: Any) -> Rewrite:
        """Reject rewriting; never changes the supplied OpenAI response."""
        from .rewrite import openai_text
        return self.rewrite(openai_text(response), **settings)

    def rewrite_anthropic(self, response: Any, **settings: Any) -> Rewrite:
        """Reject rewriting; never changes the supplied Anthropic message."""
        from .rewrite import anthropic_text
        return self.rewrite(anthropic_text(response), **settings)

    def _run(self, model: Callable[..., Any], ids: list[int], *, max_tokens: int,
             condition: str, output: str | Path | None,
             backend: Any = None, cache_factory: Any = None,
             cancel_event: Event | None = None, constraint: Any = None) -> Generation:
        from ._engine.research.keyprint_candidate_v3_caller import DurableJournal
        directory = Path(tempfile.mkdtemp(prefix="keyprint-")) if output is None else Path(output)
        if output is not None:
            directory.mkdir(mode=0o700)
        reservations: dict[str, int] = {}
        cancelled = False

        def reserve(action: str, metadata: dict[str, Any]) -> None:
            nonlocal cancelled
            if action in ("cache_creation", "model_forward", "sample", "after_grammar_mask"):
                # Never interrupt a random draw or the token commit it belongs
                # to. The frozen caller records the failure and closes cache.
                try:
                    check_cancellation(cancel_event)
                except _CancellationRequested:
                    cancelled = True
                    raise
            reservations[action] = reservations.get(action, 0) + 1

        with DurableJournal(directory / "journal.jsonl") as journal:
            report = self._candidate.run_response(
                model, ids, key=self._key, condition=condition,
                random_bits=secrets.randbits, journal=journal, reserve=reserve,
                max_tokens=max_tokens, max_model_calls=max_tokens + 8,
                allow_thinking=False, allow_tools=False,
                backend=backend, cache_factory=cache_factory,
                **({"constraint": constraint} if constraint is not None else {}),
            )
        report.setdefault("package_scope", "Namespaced reference port; no new model-family or scientific acceptance.")
        payload = report.get("payload", {})
        count = payload.get("committed_tokens", 0) if report["kind"] == "error" else len(payload.get("committed_token_ids", []))
        if cancelled:
            report["cancellation_requested"] = True
        report["usage"] = {"prompt_tokens": len(ids), "completion_tokens": count, "total_tokens": len(ids) + count}
        with (directory / "report.json").open("x", encoding="utf-8") as stream:
            json.dump({"report": report, "reservations": reservations}, stream, ensure_ascii=False, allow_nan=False)
        if report["kind"] == "error":
            if cancelled and payload.get("code") != "journal_failure":
                raise KeyprintCancelled(report, directory)
            raise KeyprintError(report, directory)
        return Generation(report["rendered_carriers"]["visible_text"], report, directory)
