"""Local, authenticated model playground. All figures come from SDK calls."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import secrets
import time
from typing import Callable, Literal
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .api import Keyprint, KeyprintError
from .inspection import Inspection
from .server import error

EXAMPLE = "In 60 words, explain how a seed becomes a tree to a curious adult."


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["generate", "inspect"]
    text: str = Field(min_length=1, max_length=16000)
    max_tokens: int = Field(default=192, ge=32, le=1024)


def inspect_text(model: Keyprint, text: str, control_key: bytes) -> dict:
    """Recompute complete literal diagnostics at bounded character prefixes.

    These are not per-token attribution or a calibrated detector trajectory.
    Prefix retokenization is intentional and disclosed in the interface.
    """
    def measure(value: str, key: bytes | None = None) -> Inspection:
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


def create_playground(load_model: Callable[[], Keyprint], *, token: str, output: Path,
                      port: int = 8766, max_requests: int = 128) -> FastAPI:
    if not isinstance(token, str) or not 32 <= len(token) <= 256 or not token.isascii():
        raise ValueError("playground token must be 32 to 256 ASCII characters")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    if type(max_requests) is not int or not 1 <= max_requests <= 4096:
        raise ValueError("max_requests must be between 1 and 4096")
    output = Path(output)
    if output.is_symlink():
        raise ValueError("playground output must not be a symlink")
    output.mkdir(mode=0o700, exist_ok=True)
    if os.name == "posix" and output.stat().st_mode & 0o077:
        raise ValueError("playground output must be private (mode 700)")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="keyprint-playground")
    model: list[Keyprint] = []
    control_key = Keyprint.new_key()
    descriptor = os.open(output / "control.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(control_key)
    lock = asyncio.Lock()
    records: dict[str, tuple[str, JSONResponse]] = {}
    attempts: set[asyncio.Task] = set()
    latest: dict = {}
    last_attempt: dict = {}
    progress: dict | None = None
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @asynccontextmanager
    async def lifespan(app):
        try:
            model.append(await asyncio.get_running_loop().run_in_executor(worker, load_model))
            yield
        finally:
            if attempts:
                await asyncio.gather(*attempts, return_exceptions=True)
            worker.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.headers.get("host") not in hosts:
            return error("Use the printed localhost address", 403)
        origin = request.headers.get("origin")
        if origin and origin != "http://" + request.headers["host"]:
            return error("Cross-origin requests are not allowed", 403)
        if request.url.path.startswith("/api/") and not secrets.compare_digest(
                request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()):
            return error("Open the complete session URL printed in your terminal", 401)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})
        return response

    @app.get("/")
    async def index():
        return HTMLResponse(files("keyprint").joinpath("web/index.html").read_text())

    @app.get("/app.{extension}")
    async def asset(extension: str):
        if extension not in ("js", "css"):
            return Response(status_code=404)
        return Response(files("keyprint").joinpath(f"web/app.{extension}").read_text(),
                        media_type="text/javascript" if extension == "js" else "text/css")

    @app.get("/reader.js")
    async def reader():
        return Response(files("keyprint").joinpath("web/reader.js").read_text(), media_type="text/javascript")

    @app.get("/api/session")
    async def session():
        return {"prompt": latest.get("prompt", EXAMPLE), "latest": latest.get("result"), "identity": model[0].identity,
                "edited": latest.get("edited"),
                "running": lock.locked(), "last_attempt": last_attempt.copy() or None,
                "scope": "Live local generation and uncalibrated literal diagnostics"}

    @app.get("/api/progress")
    async def current_progress():
        # Worker replaces the snapshot atomically; no model state crosses
        # threads. This reports real stages, not invented completion percent.
        snapshot = progress
        return ({"active": True, "stage": snapshot["stage"],
                 "seconds": time.perf_counter() - snapshot["started"]}
                if snapshot else {"active": False})

    def execute(params: Experiment, run: Path) -> dict:
        nonlocal progress
        started = time.perf_counter()
        def stage(name: str):
            nonlocal progress
            progress = {"stage": name, "started": started}
        try:
            if params.action == "inspect":
                stage("inspecting_edit")
                return {"inspection": inspect_text(model[0], params.text, control_key),
                        "seconds": time.perf_counter() - started}
            outputs = {}
            for condition in ("ordinary", "marked"):
                stage("generating_" + condition)
                generation_start = time.perf_counter()
                result = model[0].generate(params.text, max_tokens=params.max_tokens,
                                           condition=condition, output=run / condition)
                generation_seconds = time.perf_counter() - generation_start
                stage("inspecting_" + condition)
                inspection_start = time.perf_counter()
                inspection = inspect_text(model[0], result.text, control_key)
                payload = result.report.get("payload", result.report)
                outputs[condition] = {"text": result.text, "usage": result.report.get("usage"),
                                      "completion": payload.get("completion"), "inspection": inspection,
                                      "timing": {"generation_seconds": generation_seconds,
                                                 "inspection_seconds": time.perf_counter() - inspection_start}}
            return {"outputs": outputs, "max_tokens": params.max_tokens, "seconds": time.perf_counter() - started,
                    "independent_randomness": True, "calibrated": False}
        finally:
            progress = None

    @app.post("/api/experiment")
    async def experiment(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            return error("Use application/json", 415)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 65536:
                return error("Request exceeds 64 KiB", 413)
        try:
            params = Experiment.model_validate_json(bytes(raw))
            if not params.text.strip() or (params.action == "generate" and len(params.text) > 6000):
                raise ValueError("empty text")
        except (ValidationError, ValueError):
            return error("Use 1–6000 characters for prompts, 1–16000 for inspection, and a token cap from 32 to 1024", 400)
        identity = request.headers.get("idempotency-key", "")
        if not identity or len(identity) > 128 or not identity.isascii():
            return error("An idempotency key is required", 400)
        digest = hashlib.sha256(json.dumps(params.model_dump(), sort_keys=True).encode()).hexdigest()
        if identity in records:
            prior_digest, response = records[identity]
            return response if prior_digest == digest else error("Request ID already used for different input", 409)
        if lock.locked():
            return error("An experiment is running; no new work started", 503)
        if len(records) >= max_requests:
            return error("Session limit reached; restart the playground for a new session", 429)
        await lock.acquire()
        run_id = uuid.uuid4().hex
        run = output / run_id
        records[identity] = (digest, error("Attempt running; repeat the same request ID to recover its result", 409, run_id))
        last_attempt.update(run_id=run_id, action=params.action, http_status=None,
                            request=params.model_dump())

        async def finish_attempt() -> JSONResponse:
            response = error("Experiment failed. Inspect the private run folder; no automatic retry was made.", 500, run_id)
            try:
                run.mkdir(mode=0o700)
                (run / "request.json").write_text(json.dumps(params.model_dump(), ensure_ascii=False))
                try:
                    result = await asyncio.get_running_loop().run_in_executor(worker, execute, params, run)
                except Exception:
                    pass
                else:
                    result["run_id"] = run_id
                    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
                    response = JSONResponse(result)
                (run / "status.json").write_text(json.dumps({"http_status": response.status_code}))
                if response.status_code == 200 and params.action == "generate":
                    latest.update(prompt=params.text, result=result, edited=None)
                elif response.status_code == 200 and params.action == "inspect":
                    latest["edited"] = {"text": params.text, "result": result}
            except Exception:
                response = error("Experiment could not be recorded. Inspect private artifacts; no automatic retry.", 500, run_id)
            finally:
                records[identity] = (digest, response)
                last_attempt.update(http_status=response.status_code)
                lock.release()
            return response

        task = asyncio.create_task(finish_attempt())
        attempts.add(task)
        task.add_done_callback(attempts.discard)
        return await asyncio.shield(task)

    return app
