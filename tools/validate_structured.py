"""Actual local JSON inference, typed clients, caps and independently replayed masks.

Use an installed wheel with transformers, structured, server and clients extras.
Keep the output directory private; only public/ is exportable. This tests format
and integration, not semantic accuracy, watermark power or production readiness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from pydantic import BaseModel, ConfigDict
from keyprint import Keyprint
from keyprint.structured import JsonConstraint
from validate_compatibility import screens, write_report


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    count: int
    enabled: bool


def clients(candidate, output, prompt):
    import uvicorn
    from openai import OpenAI
    from anthropic import Anthropic
    from keyprint.server import create_app
    token = secrets.token_hex(32)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(lambda: candidate, api_key=token,
                          output=output / "http"), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("local structured server did not start")
            time.sleep(.05)
        requests = {}
        with OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=token, max_retries=0, timeout=180) as client:
            params = dict(model="keyprint", messages=[{"role": "user", "content": prompt}],
                          max_completion_tokens=128, response_format=Record,
                          extra_headers={"Idempotency-Key": "typed-openai"})
            response = client.chat.completions.parse(**params)
            replay = client.chat.completions.parse(**params)
            assert response.model_dump() == replay.model_dump()
            parsed = response.choices[0].message.parsed
            assert isinstance(parsed, Record)
            requests["openai"] = {"text": response.choices[0].message.content,
                "parsed": parsed.model_dump(), "request_id": response.id, "exact_replay": True}
        with Anthropic(base_url=f"http://127.0.0.1:{port}", api_key=token, max_retries=0, timeout=180) as client:
            params = dict(model="keyprint", messages=[{"role": "user", "content": prompt}],
                          max_tokens=128, output_format=Record,
                          extra_headers={"Idempotency-Key": "typed-anthropic"})
            response = client.messages.parse(**params)
            replay = client.messages.parse(**params)
            assert response.model_dump() == replay.model_dump()
            assert isinstance(response.parsed_output, Record)
            requests["anthropic"] = {"text": response.content[0].text,
                "parsed": response.parsed_output.model_dump(), "request_id": response.id, "exact_replay": True}
        assert len(list((output / "http").glob("*/report.json"))) == 2
        for value in requests.values():
            private = json.loads((output / "http" / value["request_id"] / "report.json").read_text())
            assert private["text"] == value["text"] and private["structured_output"]["schema_validated"]
        return requests
    finally:
        server.should_exit = True
        thread.join(timeout=180)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("local structured server did not stop")


def audit(candidate, output):
    total = masks = requests = 0
    for path in sorted(output.rglob("report.json")):
        report = json.loads(path.read_text())
        assert report["identity"] == candidate.identity and report["kind"] == "generation_trace"
        events, previous = [], "0" * 64
        for sequence, line in enumerate(path.with_name("journal.jsonl").read_bytes().splitlines(keepends=True)):
            row = json.loads(line)
            assert row["sequence"] == sequence and row["previous_sha256"] == previous
            previous = hashlib.sha256(line).hexdigest()
            events.append(row["event"])
        tokens = [e["token_id"] for e in events if e["phase"] == "committed"]
        assert tokens == report["committed_token_ids"]
        assert events[0]["identity"] == candidate.identity
        assert events[0]["condition"] == report["condition"]
        assert events[-1] == {"phase": "complete", "completion": report["completion"]}
        assert report["usage"]["completion_tokens"] == report["model_calls"] == len(tokens)
        assert candidate._backend.binding.render(tokens) == report["text"]
        if "structured_output" in report:
            constraint = JsonConstraint(report["structured_output"]["identity"]["schema"],
                                        candidate._backend.tokenizer, candidate._backend.binding)
            assert constraint.identity == report["structured_output"]["identity"] == events[0]["constraint"]
            allowed = None
            for event in events:
                if event["phase"] == "grammar_mask":
                    allowed = constraint.allowed()
                    assert event["allowed_sha256"] == hashlib.sha256(allowed.tobytes()).hexdigest()
                    assert event["allowed_count"] == int(allowed.sum())
                    masks += 1
                elif event["phase"] == "committed":
                    assert allowed is not None and allowed[event["token_id"]]
                    constraint.commit(event["token_id"])
                    allowed = None
            result = constraint.finish(report["text"], report["completion"])
            assert all(report["structured_output"][k] == v for k, v in result.items())
        total += len(tokens)
        requests += 1
    return {"status": "pass", "requests": requests, "tokens": total, "replayed_masks": masks,
            "scope": "Exact mask/commit and format verification; not quality or detection acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(key)
    base_case = next(c for c in json.loads(Path(__file__).with_name("inference_cases.json").read_text()) if c["id"] == "json")
    cases = [{**base_case, "id": "unconstrained-json"},
             {**base_case, "json_schema": Record.model_json_schema()},
             {"id": "unicode", "max_tokens": 128,
              "prompt": 'Return only JSON with greeting equal to "Bonjour Maya" and cities equal to ["北京", "Zürich"].',
              "json_schema": {"type": "object", "properties": {"greeting": {"type": "string"},
                "cities": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
                "required": ["greeting", "cities"], "additionalProperties": False},
              "expected_json": {"greeting": "Bonjour Maya", "cities": ["北京", "Zürich"]},
              "review": "Check literal multilingual values; grammar constrains types, not these values."},
             {"id": "negation", "max_tokens": 128,
              "prompt": 'Extract a JSON decision from: "Do not delete the backup until Maya confirms the restore succeeded." Use keys "unconditional_delete_allowed" (boolean) and "prerequisite" (string).',
              "json_schema": {"type": "object", "properties": {"unconditional_delete_allowed": {"type": "boolean"},
                "prerequisite": {"type": "string"}}, "required": ["unconditional_delete_allowed", "prerequisite"],
                "additionalProperties": False}, "review": "Must be false and retain Maya confirming successful restoration."}]
    report = {"backend": "Transformers CPU float32 / JSON grammar", "cases": cases,
        "runs": [], "clients": {}, "engineering_failures": [], "status": "started",
        "scope": __doc__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_report(args.output, report)
    try:
        candidate = Keyprint.from_transformers(args.model, key=key)
        report["identity"] = candidate.identity
        for i, case in enumerate(cases):
            for condition in (("ordinary", "marked") if i % 2 == 0 else ("marked", "ordinary")):
                result = candidate.generate(case["prompt"], condition=condition, max_tokens=case["max_tokens"],
                    output=args.output / (case["id"] + "-" + condition), json_schema=case.get("json_schema"))
                report["runs"].append({"case": case["id"], "condition": condition, "text": result.text,
                    "completion": result.report["completion"], "usage": result.report["usage"],
                    "structured_output": result.report.get("structured_output"),
                    "screens": screens(case, result.text, result.report["completion"])})
                write_report(args.output, report)
                print(case["id"], condition, result.report["completion"], flush=True)
        capped = candidate.generate(base_case["prompt"], max_tokens=1, json_schema=Record.model_json_schema(),
                                    output=args.output / "token-cap")
        assert capped.report["structured_output"]["status"] == "incomplete"
        report["token_cap"] = {"text": capped.text, "structured_output": capped.report["structured_output"]}
        report["clients"] = clients(candidate, args.output, base_case["prompt"])
        report["integrity"] = audit(candidate, args.output)
        assert report["integrity"]["requests"] == 11
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "failed"
        report["engineering_failures"].append(type(exc).__name__)
        raise
    finally:
        write_report(args.output, report)


if __name__ == "__main__":
    main()
