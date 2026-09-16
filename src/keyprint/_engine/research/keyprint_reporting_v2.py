"""Proposed v2 reporting boundary around the unchanged literal diagnostic.

Integration contract:
* ``score_literal_report(candidate, text, key)`` calls a supplied raw candidate
  exactly once; ``candidate`` must not already route through this wrapper.
* A v2 public ``score_literal`` method should return this complete report. CLI,
  browser and exported reports must preserve ``interpretation``, ``power`` and
  ``error_rates`` beside the diagnostic. Do not extract a score as a verdict.
* Invalid arguments, contradictory calibration/verdict metadata and malformed
  diagnostics raise. UI callers must display an unavailable result on exceptions,
  never a negative detector result. No retry, key discovery or model load occurs.
* This module makes no claim about fixed-threshold research reports, generation
  receipts or existing SDK/browser surfaces. Universal A22/H06 acceptance still
  requires inventory, integration and independent verification of all v2 surfaces.

No change to scientific probabilities, keyed bits, reference tails or thresholds.
No token cutoff is asserted to establish power; every result remains insufficient
for a calibrated provenance conclusion, including empty and very short inputs.
"""
from __future__ import annotations

import math
import re
from typing import Any, Protocol


REPORT_SCHEMA = "keyprint.literal-report.v2"
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
INTERPRETATION = (
    "This is an uncalibrated keyed diagnostic, not a detector verdict. "
    "It does not identify an author, owner or AI provider. Matching-key evidence "
    "from a separately calibrated detector could concern involvement in a marked "
    "generation process, including editing or translation, rather than original "
    "authorship. False positives and missed marks are possible. Short or empty "
    "text may contain too little evidence; longer text does not by itself establish "
    "power or calibration. A weak result does not establish human authorship."
)


class LiteralCandidate(Protocol):
    def score_literal(self, text: str, key: bytes) -> dict[str, Any]: ...


def _count(raw: dict[str, Any], name: str) -> int | None:
    value = raw.get(name)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"invalid diagnostic count: {name}")
    return value


def report_literal_diagnostic(raw: dict[str, Any]) -> dict[str, Any]:
    """Wrap a diagnostic without passing through arbitrary labels or nested data.

    Numeric diagnostic values are preserved. Reference-tail availability and
    calibrated inference are separate. Unknown fields (including any nested raw
    verdicts, prose labels, inferred ownership or probabilities) are not exported.
    """
    if type(raw) is not dict:
        raise TypeError("literal diagnostic must be a dict")
    if raw.get("deployment_calibrated") is not False or raw.get("verdict", object()) is not None:
        raise ValueError("report adapter requires an uncalibrated null-verdict diagnostic")
    available = raw.get("status") == "available"
    if raw.get("status") not in ("available", "unavailable"):
        raise ValueError("unknown literal diagnostic availability")
    identity: dict[str, Any] = {}
    for name in ("runtime_profile_sha256", "score_namespace_sha256", "key_commitment", "text_sha256"):
        value = raw.get(name)
        if type(value) is not str or not HEX_DIGEST.fullmatch(value):
            raise ValueError(f"invalid diagnostic digest: {name}")
        identity[name] = value
    limit = _count(raw, "runtime_max_steps")
    if limit is None or limit == 0:
        raise ValueError("missing positive diagnostic runtime bound")
    identity["runtime_max_steps"] = limit
    events, ones, trials = (_count(raw, name) for name in ("events", "ones", "trials"))
    counts = (events, ones, trials)
    if any(value is None for value in counts) and not all(value is None for value in counts):
        raise ValueError("diagnostic counts must be complete or all absent")
    if events is not None and events > limit:
        raise ValueError("diagnostic events exceed declared runtime bound")
    if available and (events is None or events == 0 or ones is None or trials is None):
        raise ValueError("available statistic requires eligible-event counts")
    if not available and any(value not in (None, 0) for value in counts):
        raise ValueError("unavailable statistic cannot contain positive event counts")
    if trials is not None and (events is None or trials != 30 * events):
        raise ValueError("diagnostic event/layer counts disagree")
    if ones is not None and (trials is None or ones > trials):
        raise ValueError("diagnostic bit count exceeds trials")
    p = raw.get("random_key_null_p")
    if available:
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid random-key reference tail")
    elif p is not None:
        raise ValueError("unavailable diagnostic cannot expose a reference tail")
    floor = raw.get("underflow_floor", False)
    if type(floor) is not bool:
        raise ValueError("invalid reference-tail underflow flag")
    if not available and floor:
        raise ValueError("unavailable statistic cannot have an underflowed reference tail")
    trace_digest = raw.get("trace_sha256")
    if trace_digest is not None and (type(trace_digest) is not str or not HEX_DIGEST.fullmatch(trace_digest)):
        raise ValueError("invalid trace digest")
    no_events = not available and events == 0
    # The two availability labels are intentionally specific to the statistic.
    # Do not relay raw status="available" without the inference limitations.
    diagnostic = {
        "kind": "whole_carrier_random_key_reference_statistic",
        "availability": "statistic_available" if available else "statistic_unavailable",
        "unavailable_reason": None if available else "no_eligible_events" if no_events else "input_not_scorable",
        "events": events, "ones": ones, "trials": trials,
        "random_key_reference_tail": {
            "value": p,
            "label": "Tail probability under the random-key reference model",
            "is_authorship_probability": False,
            "is_empirical_false_positive_rate": False,
            "underflow_floor": floor,
        },
        "identity": identity,
        "trace_sha256": trace_digest,
    }
    return {
        "schema": REPORT_SCHEMA,
        "scope": "current SDK whole-carrier literal diagnostic reporting only",
        "verdict": None,
        "attribution": {"status": "not_established", "author": None, "provider": None},
        "ownership": {"status": "not_established", "owner": None},
        "calibration": {"status": "unavailable", "deployment_calibrated": False, "threshold": None},
        "power": {
            "status": "not_established",
            "evidence_sufficiency": "insufficient_for_calibrated_conclusion",
            "reason": "no_eligible_events" if no_events else "no_validated_power_assessment",
            "minimum_length_guarantee": None,
            "short_text_warning": "Short or empty text may contain too little evidence; no token cutoff has been validated.",
        },
        "error_rates": {
            "status": "unknown_for_this_input_and_operating_setting",
            "false_positive_rate": None,
            "miss_rate": None,
            "false_positives_possible": True,
            "missed_marks_possible": True,
        },
        "interpretation": INTERPRETATION,
        "diagnostic": diagnostic,
    }


def score_literal_report(candidate: LiteralCandidate, text: str, key: bytes) -> dict[str, Any]:
    """One diagnostic call using caller-owned inputs, with no automatic retry."""
    if type(text) is not str:
        raise TypeError("carrier text must be str")
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    return report_literal_diagnostic(candidate.score_literal(text, key))
