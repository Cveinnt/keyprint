"""Authenticated local Chat Completions subset; not a production inference server."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from typing import Callable, Literal
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .api import Keyprint


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["user"]
    content: str = Field(min_length=1, max_length=16000)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: Literal["keyprint"]
    messages: list[Message] = Field(min_length=1, max_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=1024)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=1024)
    stream: Literal[False] = False
    n: Literal[1] = 1

    @field_validator("stream", "n", mode="before")
    @classmethod
    def exact_scalar(cls, value, info):
        expected = bool if info.field_name == "stream" else int
        if type(value) is not expected:
            raise ValueError("wrong scalar type")
        return value


def error(message: str, status: int, request_id: str | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "keyprint_error", "code": str(status)},
                         "request_id": request_id}, status_code=status,
                        headers={"x-should-retry": "false"})


def create_app(load_model: Callable[[], Keyprint], *, api_key: str, output: Path,
               max_requests: int = 256) -> FastAPI:
    """One process, one model worker, finite requests; secret keys stay local.

    Requests are retained even after a client disconnect. Idempotency applies
    only in this process lifetime. No retry, streaming, tools or logprobs.
    """
    if not isinstance(api_key, str) or not 32 <= len(api_key) <= 256 or not api_key.isascii():
        raise ValueError("API token must be 32 to 256 ASCII characters, distinct from the watermark key")
    if type(max_requests) is not int or not 1 <= max_requests <= 4096:
        raise ValueError("max_requests must be between 1 and 4096")
    output = Path(output)
    if output.is_symlink():
        raise ValueError("server output must not be a symlink")
    output.mkdir(mode=0o700, exist_ok=True)
    if os.name == "posix" and output.stat().st_mode & 0o077:
        raise ValueError("server output directory must be private (mode 700)")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="keyprint-model")
    lock = asyncio.Lock()
    records: dict[str, tuple[str, JSONResponse]] = {}
    attempts: set[asyncio.Task] = set()
    model: list[Keyprint] = []

    @asynccontextmanager
    async def lifespan(app):
        try:
            model.append(await asyncio.get_running_loop().run_in_executor(worker, load_model))
            yield
        finally:
            # Accepted work belongs to the service, not to a socket. Graceful
            # shutdown drains it before disposing the model's owning worker.
            if attempts:
                await asyncio.gather(*attempts, return_exceptions=True)
            worker.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Keyprint local preview", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = ("Bearer " + api_key).encode()
        actual = request.headers.get("authorization", "").encode()
        if not secrets.compare_digest(actual, expected):
            return error("Invalid local API token", 401)
        return await call_next(request)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "keyprint", "object": "model", "created": 0,
                                            "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            return error("Use application/json", 415)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 131072:
                return error("Request exceeds 128 KiB", 413)
        try:
            params = CompletionRequest.model_validate_json(bytes(raw))
        except ValidationError:
            return error("Supported: model=keyprint, one user text message, one token cap, stream=false, n=1. Other options are rejected.", 400)
        if params.max_tokens is not None and params.max_completion_tokens is not None:
            return error("Choose one token cap", 400)
        cap = params.max_completion_tokens or params.max_tokens or 128
        request_id = "chatcmpl-" + uuid.uuid4().hex
        idempotency = request.headers.get("idempotency-key", request_id)
        if not idempotency or len(idempotency) > 128 or not idempotency.isascii():
            return error("Invalid idempotency key", 400)
        digest = hashlib.sha256(json.dumps(params.model_dump(), sort_keys=True).encode()).hexdigest()
        previous = records.get(idempotency)
        if previous:
            return previous[1] if previous[0] == digest else error("Idempotency key was used for another request", 409)
        if lock.locked():
            return error("Model is busy; no generation started for this request", 503)
        if len(records) >= max_requests:
            return error("Session request limit reached; inspect retained artifacts before restarting", 429)
        # Acquisition completes without suspension while unlocked. Record and
        # own the attempt before yielding to any model work or HTTP response.
        await lock.acquire()
        records[idempotency] = (digest, error("Attempt running; repeat the same request ID to recover its result", 409, request_id))

        async def execute() -> JSONResponse:
            response = error("Generation failed; inspect the server's private report. No automatic retry.", 500, request_id)
            try:
                with (output / (request_id + "-started.json")).open("x") as stream:
                    json.dump({"request_id": request_id, "request_sha256": digest, "state": "started"}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    result = await asyncio.get_running_loop().run_in_executor(worker, lambda: model[0].generate(
                        params.messages[0].content, max_tokens=cap, output=output / request_id))
                except ValueError:
                    response = error("Prompt is outside this model's supported input contract", 400, request_id)
                except Exception:
                    pass
                else:
                    payload = result.report.get("payload", result.report)
                    usage = result.report.get("usage")
                    response = JSONResponse({"id": request_id, "object": "chat.completion",
                        "created": int(time.time()), "model": "keyprint",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                                     "finish_reason": "stop" if payload.get("completion") == "eos" else "length",
                                     "logprobs": None}], "usage": usage,
                        "keyprint": {"mode": "local_marked_generation", "hosted_provider": False,
                                     "detection_calibrated": False}}, headers={"x-request-id": request_id})
                with (output / (request_id + "-finished.json")).open("x") as stream:
                    json.dump({"request_id": request_id, "http_status": response.status_code}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                response = error("Attempt could not be recorded; inspect private artifacts. No automatic retry.", 500, request_id)
            finally:
                records[idempotency] = (digest, response)
                lock.release()
            return response

        task = asyncio.create_task(execute())
        attempts.add(task)
        task.add_done_callback(attempts.discard)
        return await asyncio.shield(task)

    return app
