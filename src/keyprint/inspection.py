"""Small, typed view of literal diagnostics; never a detection verdict."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Inspection:
    """Counts from replaying visible text under one key and tokenizer profile.

    ``fraction`` is the observed fraction of one bits, not confidence or an
    authorship probability. It is None when there are no eligible bits.
    The original scientific report is retained without relabeling its evidence.
    """

    events: int | None
    ones: int | None
    trials: int | None
    report: dict[str, Any]

    @property
    def fraction(self) -> float | None:
        return self.ones / self.trials if self.trials and self.ones is not None else None

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> Inspection:
        if report.get("kind") != "literal_diagnostic":
            raise ValueError("Expected a literal diagnostic report")
        if "payload" in report:
            payload = report["payload"]
            counts = [payload.get(name) for name in ("events", "ones", "trials")]
        else:
            counts = [report.get(name) for name in ("eligible_events", "one_bits", "total_bits")]
        if any(value is None for value in counts):
            if not all(value is None for value in counts):
                raise ValueError("Diagnostic counts must be all available or all unavailable")
        elif (any(type(value) is not int or value < 0 for value in counts)
              or counts[1] > counts[2]):
            raise ValueError("Invalid diagnostic counts")
        return cls(*counts, report=report)
