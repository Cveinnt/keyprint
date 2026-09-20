"""Shared portable-profile sampling, journals and failure accounting.

Backends supply prompt encoding, a request-local forward context and exact byte
rendering. No backend sampler runs after Keyprint selects a token.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import tempfile

from ..sampling import sparse_sample, sparse_softmax
from ..errors import InputLimitError


class PortableGeneration:
    def score(self, text: str, key: bytes) -> dict:
        from .._engine.legacy._impl.research.grouped_canonical_prototype import replay_events
        if not isinstance(text, str) or len(text) > 16000:
            raise ValueError("text must be a string of at most 16000 characters")
        ids = self.encode_literal(text)
        if any(self.binding.pieces[i] is None for i in ids) or self.binding.render(ids) != text:
            raise ValueError("literal text is not exactly replayable with this tokenizer binding")
        events = replay_events(self.profile, key, ids)
        bits = [bit for event in events if event.eligible for bit in (event.bits or ())]
        return {"kind": "literal_diagnostic", "verdict": None, "identity": self.identity,
                "eligible_events": sum(event.eligible for event in events),
                "one_bits": sum(bits), "total_bits": len(bits),
                "interpretation": "Uncalibrated matching-key event counts on literal tokenization; no p-value or authorship verdict."}

    def generate(self, prompt: str, *, key: bytes, max_tokens: int, condition: str,
                 output: str | Path | None, cancel_event=None, json_schema=None):
        import numpy as np
        from ..api import Generation, KeyprintError, KeyprintCancelled
        from ..cancellation import check_cancellation, _CancellationRequested
        from .._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
        from .._engine.research.keyprint_candidate_v3_caller import DurableJournal
        from .._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter

        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
            raise ValueError("prompt must contain 1 to 16000 characters")
        constraint = None
        if json_schema is not None:
            constraint = self.json_constraint(json_schema)
        encoded = self.encode_prompt(prompt)
        if any(type(i) is not int or not 0 <= i < len(self.binding.pieces) for i in encoded):
            raise ValueError("prompt tokens are outside the declared model binding")
        if not encoded:
            raise ValueError("prompt tokens are empty")
        limit = min(self.context_limit, 8192)
        if len(encoded) + max_tokens > limit:
            raise InputLimitError(input_tokens=len(encoded), max_tokens=max_tokens, limit=limit)
        directory = Path(tempfile.mkdtemp(prefix="keyprint-")) if output is None else Path(output)
        if output is not None:
            directory.mkdir(mode=0o700)
        session = SparseTokenSourceSession(self.profile, key, condition=condition)
        committed: list[int] = []
        report = {"kind": "generation_trace", "verdict": None, "identity": self.identity,
                  "condition": condition, "completion": "length", "committed_token_ids": committed,
                  "model_calls": 0, "interpretation": "Experimental portable profile; no research acceptance or calibrated detection claim."}
        if constraint is not None:
            report["structured_output"] = {"identity": constraint.identity, "status": "incomplete", "schema_validated": False}
        phase = "start"
        try:
            with DurableJournal(directory / "journal.jsonl") as journal, self.inference_session() as forward:
                journal.append({"phase": "start", "identity": self.identity,
                                "condition": condition, "prompt_token_ids": encoded,
                                **({"constraint": constraint.identity} if constraint is not None else {})})
                ids = encoded
                for _ in range(max_tokens):
                    phase = "before_model_forward"
                    check_cancellation(cancel_event)
                    phase = "model_forward"
                    journal.append({"phase": phase, "index": len(committed)})
                    report["model_calls"] += 1
                    raw = forward(ids)
                    if raw.dtype != np.float32 or raw.shape != (1, len(self.binding.pieces)):
                        raise ValueError("model returned an unsupported logit shape or dtype")
                    if np.isnan(raw).any() or np.isposinf(raw).any():
                        raise ValueError("model returned invalid logits")
                    phase = "before_sample"
                    check_cancellation(cancel_event)
                    phase = "sample"
                    raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                    raw = raw.copy()
                    for index, piece in enumerate(self.binding.pieces):
                        if piece is None and index not in self.binding.eos_ids:
                            raw[0, index] = -np.inf
                    if constraint is not None:
                        phase = "grammar_mask"
                        allowed = constraint.allowed()
                        raw[0, ~allowed] = -np.inf
                        journal.append({"phase": phase, "index": len(committed),
                            "allowed_sha256": hashlib.sha256(allowed.tobytes()).hexdigest(),
                            "allowed_count": int(allowed.sum())})
                        check_cancellation(cancel_event)
                        phase = "sample"
                    filtered = stable_support_filter(raw, temperature=self.temperature, top_k=self.top_k,
                                                     mapped_vocabulary_size=len(self.binding.pieces))
                    base = sparse_softmax(filtered.filtered_logits[0])
                    prepared = session.prepare(base)
                    journal.append({"phase": "prepared", "raw_logits_sha256": raw_hash,
                                    "weights_sha256": hashlib.sha256(prepared.probabilities.tobytes()).hexdigest()})

                    def random_bits(count: int) -> int:
                        journal.append({"phase": "random_requested", "bits": count})
                        value = secrets.randbits(count)
                        journal.append({"phase": "random_returned", "bits": count, "value": value})
                        return value

                    draw = sparse_sample(prepared.probabilities, random_bits)
                    selected = draw.token_index
                    # Persist intent before any irreversible session mutation.
                    journal.append({"phase": "commit_requested", "token_id": selected, "draw": asdict(draw)})
                    session.commit(prepared, selected)
                    committed.append(selected)
                    journal.append({"phase": "committed", "token_id": selected})
                    if constraint is not None:
                        phase = "grammar_commit"
                        constraint.commit(selected)
                    if selected in self.binding.eos_ids:
                        report["completion"] = "eos"
                        break
                    ids = [selected]
                phase = "render"
                if report["completion"] == "length":
                    text, pending = self.binding.render_at_limit(committed, max_tokens)
                else:
                    text, pending = self.binding.render(committed), b""
                if not pending:
                    decoded = self.decode_tokens(committed)
                    if text != decoded:
                        raise ValueError("tokenizer rendering differs from the declared byte binding")
                report["carrier_rendering"] = [{"channel": "visible",
                    "status": "incomplete_utf8_at_token_limit" if pending else "complete_utf8",
                    "pending_utf8_hex": pending.hex(), "committed_token_ids": committed.copy()}]
                report["tokenizer_rendering_check"] = (
                    "unavailable_for_incomplete_utf8_carrier" if pending else "exact_match")
                report["literal_replay_status"] = [{"channel": "visible",
                    "availability": "unavailable" if pending else "not_measured",
                    "reason": "Token limit split a UTF-8 character; rendered prefix is not the complete sampled carrier"
                    if pending else "Literal inspection is a separate operation"}]
                report["text"] = text
                if constraint is not None:
                    phase = "structured_validation"
                    report["structured_output"].update(constraint.finish(text, report["completion"]))
                report["usage"] = {"prompt_tokens": len(encoded), "completion_tokens": len(committed),
                                   "total_tokens": len(encoded) + len(committed)}
                report["source_receipt"] = session.source_receipt()
                journal.append({"phase": "complete", "completion": report["completion"]})
        except BaseException as exc:
            report.update(kind="error", failed_phase=phase, error_type=type(exc).__name__)
            report["usage"] = {"prompt_tokens": len(encoded), "completion_tokens": len(committed),
                               "total_tokens": len(encoded) + len(committed)}
            if isinstance(exc, _CancellationRequested):
                report["cancellation_requested"] = True
            # Keep even interrupted attempts. Do not rerun or reuse the session.
            with (directory / "report.json").open("x") as stream:
                json.dump(report, stream, ensure_ascii=False, allow_nan=False)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, _CancellationRequested):
                raise KeyprintCancelled(report, directory) from exc
            raise KeyprintError(report, directory) from exc
        finally:
            session.close()
        with (directory / "report.json").open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False)
        return Generation(text, report, directory)
