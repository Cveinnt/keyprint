"""Reusable paired generation and prefix diagnostics, without a web dependency."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib.resources import files
import json
from pathlib import Path
import tempfile
from threading import Event
import time
from typing import Callable

from .api import Keyprint, KeyprintError
from .cancellation import check_cancellation
from .inspection import Inspection

def inspect_text(model: Keyprint, text: str, control_key: bytes, cancel_event: Event | None = None) -> dict:
    """Recompute complete literal diagnostics at bounded character prefixes.

    These are not per-token attribution or a calibrated detector trajectory.
    Prefix retokenization is intentional and disclosed in the interface.
    """
    def measure(value: str, key: bytes | None = None) -> Inspection:
        check_cancellation(cancel_event)
        try:
            return model.inspect(value, **({"key": key} if key is not None else {}))
        except (KeyprintError, ValueError):
            # A token-boundary or replay failure is not a zero signal, and must
            # not discard a successfully generated response.
            return Inspection(None, None, None, {"kind": "literal_diagnostic", "verdict": None,
                "availability": "unavailable", "reason": "Literal replay unavailable for this text and profile"})

    positions = sorted({len(text), *(round(len(text) * i / 16) for i in range(1, 16))} - {0})
    series = []
    for end in positions:
        matching = measure(text[:end])
        control = measure(text[:end], key=control_key)
        series.append({"characters": end, "matching": matching.fraction,
                       "control": control.fraction})
    # The final prefix is the complete text. Reuse those exact reports rather
    # than replaying it twice more after plotting it.
    if not positions:
        matching, control = measure(text), measure(text, key=control_key)
    return {"series": series, "events": matching.events, "ones": matching.ones,
            "trials": matching.trials, "fraction": matching.fraction,
            "control_fraction": control.fraction, "report": matching.report,
            "control_report": control.report, "calibrated": False, "verdict": None}



def compare(model: Keyprint, prompt: str, *, max_tokens: int = 192,
            control_key: bytes | None = None, output: Path | None = None,
            cancel_event: Event | None = None,
            on_stage: Callable[[str], None] | None = None,
            outputs: dict | None = None) -> Comparison:
    """Two independent generations plus literal prefix diagnostics.

    No rewriting, verdict, or semantic guarantee. Both conditions use the same
    model, prompt and cap; randomness is independent. Journals stay private.
    """
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
        raise ValueError("prompt must contain 1 to 16000 characters and not be blank")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be between 1 and 1024")
    if cancel_event is not None and not isinstance(cancel_event, Event):
        raise TypeError("cancel_event must be a threading.Event or None")
    control_key = Keyprint.new_key() if control_key is None else control_key
    if type(control_key) is not bytes or len(control_key) != 32:
        raise ValueError("control_key must be exactly 32 bytes")
    check_cancellation(cancel_event)
    run = Path(output) if output is not None else Path(tempfile.mkdtemp(prefix="keyprint-compare-"))
    if run.is_symlink():
        raise ValueError("comparison output must not be a symlink")
    run.mkdir(mode=0o700, parents=True, exist_ok=True)
    if run.stat().st_mode & 0o077:
        raise ValueError("comparison output must be private (mode 700)")
    # Retain the independent key for reproducibility, never in exported data.
    with (run / "comparison-control.key").open("xb") as stream:
        stream.write(control_key)
    (run / "comparison-control.key").chmod(0o600)
    started = time.perf_counter()
    outputs = {} if outputs is None else outputs
    for condition in ("ordinary", "marked"):
        if on_stage:
            on_stage("generating_" + condition)
        check_cancellation(cancel_event)
        generation_start = time.perf_counter()
        result = model.generate(prompt, max_tokens=max_tokens, condition=condition,
                                output=run / condition, cancel_event=cancel_event, trace=True)
        generation_seconds = time.perf_counter() - generation_start
        if on_stage:
            on_stage("inspecting_" + condition)
        inspection_start = time.perf_counter()
        inspection = inspect_text(model, result.text, control_key, cancel_event)
        payload = result.report.get("payload", result.report)
        outputs[condition] = {"text": result.text, "usage": result.report.get("usage"),
            "completion": payload.get("completion"), "inspection": inspection,
            "trace": [asdict(step) for step in result.trace] if result.trace is not None else None,
            "timing": {"generation_seconds": generation_seconds,
                       "inspection_seconds": time.perf_counter() - inspection_start}}
    return Comparison(prompt, {"outputs": outputs, "max_tokens": max_tokens,
        "backend": {"MLXModel": "mlx", "TransformersModel": "transformers",
                    "LlamaCppModel": "llama_cpp"}.get(type(getattr(model, "_backend", None)).__name__),
        "seconds": time.perf_counter() - started, "independent_randomness": True,
        "calibrated": False})


@dataclass(frozen=True)
class Comparison:
    """A real paired run. Export contains text and measurements, never keys/journals."""
    prompt: str
    result: dict

    @property
    def ordinary(self) -> str:
        return self.result["outputs"]["ordinary"]["text"]

    @property
    def marked(self) -> str:
        return self.result["outputs"]["marked"]["text"]

    def to_dict(self) -> dict:
        """JSON-ready data for custom visuals; strips raw reports and local paths.

        Prompts and generated text are included. Review their contents before
        sharing. Fractions are observations, not detection confidence.
        """
        outputs = {}
        for condition in ("ordinary", "marked"):
            row = self.result["outputs"][condition]
            inspection = row["inspection"]
            outputs[condition] = {
                "text": row["text"], "completion": row.get("completion"),
                "usage": {k: row.get("usage", {}).get(k) for k in
                    ("prompt_tokens", "completion_tokens", "total_tokens")} if row.get("usage") else None,
                "timing": {k: row.get("timing", {}).get(k) for k in
                    ("generation_seconds", "inspection_seconds")},
                "inspection": {k: inspection.get(k) for k in
                    ("events", "ones", "trials", "fraction", "control_fraction")}
            }
            # Trace is an allowlisted projection, never the raw generation journal.
            if row.get("trace") is not None:
                outputs[condition]["trace"] = [{k: step[k] for k in
                    ("index", "token_id", "bytes_hex", "text", "start", "end", "kind")}
                    for step in row["trace"]]
            outputs[condition]["inspection"].update(calibrated=False, verdict=None,
                series=[{k: point.get(k) for k in ("characters", "matching", "control")}
                        for point in inspection["series"]])
        data = {"schema": "keyprint-comparison-v1", "prompt": self.prompt,
            "experiment": {"outputs": outputs,
                "backend": self.result.get("backend") if self.result.get("backend") in
                    ("mlx", "transformers", "llama_cpp") else None,
                "seconds": self.result["seconds"],
                "max_tokens": self.result["max_tokens"], "independent_randomness": True,
                "calibrated": False}}
        return json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))

    def export(self, directory: str | Path) -> Path:
        """Write a static replay, using the same viewer as the live playground.

        Destination must not exist. Host locally with `python -m http.server`;
        no model/server dependencies are required to view this recorded run.
        Does not upload or publish anything. Review text before sharing.
        """
        data = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        web = files("keyprint").joinpath("web")
        for name in ("index.html", "app.js", "app.css", "reader.js", "trace.js", "gallery.js", "favicon.svg"):
            content = web.joinpath(name).read_text()
            if name == "index.html":
                content = content.replace('<body>', '<body data-mode="replay">')
                content = content.replace('the local playground', 'recorded playground')
            (target / name).write_text(content)
        (target / "replay.json").write_text(data)
        (target / "README.txt").write_text(
            "Keyprint recorded SDK run. Serve this directory with python -m http.server.\n"
            "Prompt and output text are included; review before sharing.\n"
            "No watermark keys or private journals are included. No model runs in replay.\n"
            "Fractions are uncalibrated observations, not detection confidence.\n")
        return target / "index.html"


def export_gallery(comparisons: dict[str, Comparison], directory: str | Path, *,
                   notes: dict[str, str] | None = None) -> Path:
    """Export up to twelve named, real comparisons in one static viewer.

    No generation or upload occurs. All recordings are retained in insertion
    order, without ranking by signal or quality. Prompts, outputs and titles
    become shareable content; review them before publishing the directory.
    """
    if not isinstance(comparisons, dict) or not 1 <= len(comparisons) <= 12:
        raise ValueError("gallery needs between 1 and 12 named comparisons")
    notes = {} if notes is None else notes
    if not isinstance(notes, dict) or any(title not in comparisons or
            not isinstance(note, str) or len(note) > 1000 for title, note in notes.items()):
        raise ValueError("gallery notes must name an included example and contain at most 1000 characters")
    recordings = []
    for title, pair in comparisons.items():
        if not isinstance(title, str) or not title.strip() or len(title) > 80:
            raise ValueError("gallery titles must contain 1 to 80 characters")
        if not isinstance(pair, Comparison):
            raise TypeError("gallery values must be Comparison instances")
        recordings.append({"title": title, "note": notes.get(title, ""), "recording": pair.to_dict()})
    # Serialize and validate before creating any files.
    payload = json.dumps({"schema": "keyprint-gallery-v1", "examples": recordings},
                         ensure_ascii=False, allow_nan=False, indent=2)
    entry = next(iter(comparisons.values())).export(directory)
    entry.write_text(entry.read_text().replace('data-mode="replay"',
                                             'data-mode="replay" data-gallery="true"'))
    (entry.parent / "gallery.json").write_text(payload)
    return entry
