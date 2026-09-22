"""Local, authenticated model playground. All figures come from SDK calls."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
from importlib.resources import files
import json
import logging
import os
from pathlib import Path
import secrets
import time
from threading import Event
from typing import Callable, Literal
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .api import Keyprint, KeyprintError, KeyprintCancelled
from .errors import InputLimitError, RewriteUnavailableError
from .cancellation import check_cancellation, _CancellationRequested
from .inspection import Inspection
from .rewrite import plain_text, protected_literals
from .server import error

EXAMPLE = "In 60 words, explain how a seed becomes a tree to a curious adult."


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["generate", "inspect", "rewrite"]
    text: str = Field(min_length=1, max_length=16000)
    max_tokens: int = Field(default=192, ge=32, le=1024)
    preserve: list[str] = Field(default_factory=list, max_length=32)


from .comparison import inspect_text, compare


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
    cancellations: dict[str, Event] = {}
    attempts: set[asyncio.Task] = set()
    latest: dict = {}
    last_attempt: dict = {}
    progress: dict | None = None
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    startup = {"status": "loading", "seconds": 0.0}
    model_identity = None
    loading_started = None

    async def initialize():
        nonlocal model_identity, loading_started
        loading_started = time.perf_counter()
        try:
            def load():
                candidate = load_model()
                return candidate, candidate.identity
            candidate, identity = await asyncio.get_running_loop().run_in_executor(worker, load)
            elapsed = time.perf_counter() - loading_started
            (output / "startup.json").write_text(json.dumps({"status": "ready", "seconds": elapsed}))
            model.append(candidate)
            model_identity = identity
            startup.update(status="ready", seconds=elapsed)
        except Exception as exc:
            startup.update(status="failed", seconds=time.perf_counter() - loading_started,
                message="The local model could not load. Check the terminal and private startup.json, fix the model path or dependencies, then restart keyprint playground.")
            try:
                (output / "startup.json").write_text(json.dumps({**startup,
                    "error_type": type(exc).__name__, "detail": str(exc)}))
            except OSError:
                pass
            logging.getLogger(__name__).exception("Local model startup failed; private artifacts: %s", output)

    @asynccontextmanager
    async def lifespan(app):
        loading = asyncio.create_task(initialize())
        try:
            yield
        finally:
            try:
                await asyncio.gather(loading, *attempts, return_exceptions=True)
                for candidate in model:
                    close = getattr(candidate, "close", None)
                    if close is not None:
                        await asyncio.get_running_loop().run_in_executor(worker, close)
            finally:
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

    @app.get("/trace.js")
    async def trace_asset():
        return Response(files("keyprint").joinpath("web/trace.js").read_text(), media_type="text/javascript")

    @app.get("/gallery.js")
    async def gallery_asset():
        return Response(files("keyprint").joinpath("web/gallery.js").read_text(), media_type="text/javascript")

    @app.get("/reader.js")
    async def reader():
        return Response(files("keyprint").joinpath("web/reader.js").read_text(), media_type="text/javascript")

    @app.get("/favicon.svg")
    async def favicon():
        return Response(files("keyprint").joinpath("web/favicon.svg").read_text(), media_type="image/svg+xml")

    @app.get("/api/session")
    async def session():
        status = startup.copy()
        if status["status"] == "loading" and loading_started is not None:
            status["seconds"] = time.perf_counter() - loading_started
        return {"prompt": latest.get("prompt", EXAMPLE), "latest": latest.get("result"), "identity": model_identity,
                "model": status,
                "edited": latest.get("edited"),
                "running": lock.locked(), "last_attempt": last_attempt.copy() or None,
                "scope": "Live local generation and uncalibrated literal diagnostics"}

    @app.get("/api/progress")
    async def current_progress():
        # Worker replaces the snapshot atomically; no model state crosses
        # threads. This reports real stages, not invented completion percent.
        snapshot = progress
        if snapshot:
            return {"active": True, "stage": snapshot["stage"],
                    "seconds": time.perf_counter() - snapshot["started"],
                    "request_id": last_attempt["request_id"],
                    "cancellation_requested": last_attempt.get("cancellation_requested", False)}
        return {"active": False}

    @app.post("/api/cancel")
    async def cancel(request: Request):
        identity = request.headers.get("idempotency-key", "")
        if not identity or len(identity) > 128 or not identity.isascii():
            return error("An idempotency key is required", 400)
        previous = records.get(identity)
        if previous is None:
            return error("That experiment has not started", 404)
        signal = cancellations.get(identity)
        if signal is None:
            return {"state": "terminal", "http_status": previous[1].status_code}
        signal.set()
        last_attempt["cancellation_requested"] = True
        return JSONResponse({"state": "cancellation_requested"}, status_code=202)

    def execute(params: Experiment, run: Path, cancellation: Event) -> dict:
        nonlocal progress
        started = time.perf_counter()
        def stage(name: str):
            nonlocal progress
            progress = {"stage": name, "started": started}
        outputs = {}
        try:
            if params.action == "inspect":
                stage("inspecting_edit")
                return {"inspection": inspect_text(model[0], params.text, control_key, cancellation),
                        "seconds": time.perf_counter() - started}
            return compare(model[0], params.text, max_tokens=params.max_tokens,
                           control_key=control_key, output=run, cancel_event=cancellation,
                           on_stage=stage, outputs=outputs).result
        except (KeyprintCancelled, _CancellationRequested):
            (run / "cancelled.json").write_text(json.dumps({
                "action": params.action, "stage": progress["stage"],
                "completed_conditions": list(outputs), "seconds": time.perf_counter() - started,
                "scope": "Stopped attempt; individual generation receipts remain private"}))
            raise
        finally:
            progress = None

    @app.post("/api/experiment")
    async def experiment(request: Request):
        if startup["status"] != "ready":
            return error(startup.get("message", "The local model is still loading; no experiment started"), 503)
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
            if params.action == "rewrite":
                plain_text(params.text)
                protected_literals(params.text, params.preserve)
            elif params.preserve:
                raise ValueError("preserve applies only to rewriting")
        except (ValidationError, ValueError):
            return error("Use 1–6000 characters for prompts, 1–8000 of prose for rewriting, or 1–16000 for inspection; token cap 32–1024. Preserved phrases must be distinct, present in the source, and used only with rewriting.", 400)
        if params.action == "rewrite":
            return error(str(RewriteUnavailableError()), 400)
        identity = request.headers.get("idempotency-key", "")
        if not identity or len(identity) > 128 or not identity.isascii():
            return error("An idempotency key is required", 400)
        request_body = params.model_dump(exclude={"preserve"} if not params.preserve else set())
        digest = hashlib.sha256(json.dumps(request_body, sort_keys=True).encode()).hexdigest()
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
        cancellation = Event()
        cancellations[identity] = cancellation
        records[identity] = (digest, error("Attempt running; repeat the same request ID to recover its result", 409, run_id))
        last_attempt.update(run_id=run_id, request_id=identity, action=params.action, http_status=None,
                            request=request_body, cancellation_requested=False)

        async def finish_attempt() -> JSONResponse:
            response = error("Experiment failed. Inspect the private run folder; no automatic retry was made.", 500, run_id)
            try:
                run.mkdir(mode=0o700)
                (run / "request.json").write_text(json.dumps(request_body, ensure_ascii=False))
                try:
                    result = await asyncio.get_running_loop().run_in_executor(worker, execute, params, run, cancellation)
                except (KeyprintCancelled, _CancellationRequested):
                    response = error("Experiment stopped. Previous completed results are unchanged.", 410, run_id)
                except InputLimitError as exc:
                    response = error(str(exc), 400, run_id)
                except Exception:
                    pass
                else:
                    result["run_id"] = run_id
                    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
                    response = JSONResponse(result)
                (run / "status.json").write_text(json.dumps({"http_status": response.status_code}))
                if response.status_code == 200 and params.action in ("generate", "rewrite"):
                    latest.update(prompt=params.text, result=result, edited=None)
                elif response.status_code == 200 and params.action == "inspect":
                    latest["edited"] = {"text": params.text, "result": result}
            except Exception:
                response = error("Experiment could not be recorded. Inspect private artifacts; no automatic retry.", 500, run_id)
            finally:
                records[identity] = (digest, response)
                cancellations.pop(identity, None)
                last_attempt.update(http_status=response.status_code)
                lock.release()
            return response

        task = asyncio.create_task(finish_attempt())
        attempts.add(task)
        task.add_done_callback(attempts.discard)
        return await asyncio.shield(task)

    return app
