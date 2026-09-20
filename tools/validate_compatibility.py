"""Actual local inference comparisons and SDK-over-HTTP integration checks.

Use an installed wheel. Private run folders contain keys/journals. Only public/
contains exportable synthetic-case reports. Lexical checks are review flags,
not semantic quality or detector acceptance. Every attempt is retained.
"""
import argparse
import difflib
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from keyprint import Keyprint


def screens(case, text, completion):
    result = {"complete": completion == "eos", "nonempty": bool(text.strip()),
              "missing_literals": [word for word in case.get("required_literals", []) if word not in text],
              "semantic_quality": "requires_review"}
    if "expected_json" in case:
        try:
            parsed = json.loads(text)
            result["exact_json"] = (parsed == case["expected_json"] and
                                    {k: type(v) for k, v in parsed.items()} ==
                                    {k: type(v) for k, v in case["expected_json"].items()})
        except (ValueError, TypeError, AttributeError):
            result["exact_json"] = False
    return result


def inspect(candidate, text, other_key):
    result = {}
    for name, key in (("matching", None), ("other", other_key)):
        try:
            measurement = candidate.inspect(text, **({"key": key} if key else {}))
            result[name] = {"fraction": measurement.fraction, "events": measurement.events,
                            "ones": measurement.ones, "trials": measurement.trials}
        except Exception as exc:
            result[name] = {"unavailable": type(exc).__name__}
    return result


def write_report(output, report):
    report["screening_failures"] = [
        {"case": row["case"], "condition": row["condition"], "screens": row["screens"]}
        for row in report["runs"] if "screens" in row and
        (not row["screens"]["complete"] or not row["screens"]["nonempty"] or
         row["screens"]["missing_literals"] or row["screens"].get("exact_json") is False)]
    report["quality_acceptance"] = "not_established; mechanical screens do not check meaning"
    public = output / "public"
    public.mkdir(exist_ok=True)
    (public / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    sections = []
    for case in report["cases"]:
        rows = sorted((r for r in report["runs"] if r["case"] == case["id"]),
                      key=lambda row: row["condition"] != "ordinary")
        columns = []
        for row in rows:
            columns.append('<article><h3>' + html.escape(row["condition"]) + '</h3><pre>' +
                           html.escape(row.get("text", "Generation failed")) + '</pre><details><summary>Measured results and review flags</summary><pre>' +
                           html.escape(json.dumps({k:v for k,v in row.items() if k != "text"}, indent=2, ensure_ascii=False)) + '</pre></details></article>')
        sections.append('<section><h2>' + html.escape(case["id"]) + '</h2><p>' + html.escape(case["prompt"]) +
                        '</p><p><em>' + html.escape(case["review"]) + '</em></p><div class="pair">' + ''.join(columns) + '</div></section>')
    client_sections = []
    for name, result in report.get("clients", {}).items():
        if "original" in result and "text" in result:
            client_sections.append('<section><h2>' + html.escape(name.replace('_', ' ')) +
                '</h2><p>Constructed provider response object; real local model rewrite. No hosted API call.</p><div class="pair">' +
                '<article><h3>Original</h3><pre>' + html.escape(result['original']) + '</pre></article>' +
                '<article><h3>Local rewrite</h3><pre>' + html.escape(result['text']) + '</pre></article></div></section>')
    (public / "comparison.html").write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Keyprint inference comparisons</title><style>body{max-width:1160px;margin:48px auto;padding:0 24px;background:#f7f5ee;color:#292923;font:18px/1.55 Georgia,serif}h1{font-size:42px}h2{font-size:28px}section{border-top:1px solid #ccc6b7;padding:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:#fffdf8;padding:22px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 Georgia,serif}details pre{font:12px/1.5 monospace}summary{cursor:pointer}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>
<h1>Actual inference. Both texts.</h1><p>Independent ordinary and marked samples. All attempts retained. Mechanical flags are not semantic approval; fractions are not detection confidence.</p>''' +
        '<p>' + html.escape(report["backend"]) + ' · ' + str(len(report["screening_failures"])) +
        ' generated outputs flagged by mechanical screens. Semantic quality remains unapproved.</p>' + ''.join(sections) + ''.join(client_sections) +
        '<section><h2>Client checks</h2><pre>' + html.escape(json.dumps(report.get("clients", {}), indent=2, ensure_ascii=False)) + '</pre></section>')


def client_check(loader, output):
    """Real OpenAI/Anthropic clients over TCP with actual local model inference."""
    import uvicorn
    from openai import OpenAI, APIStatusError
    from anthropic import Anthropic, APIStatusError as AnthropicStatusError
    from keyprint.server import create_app
    token = secrets.token_hex(32)
    app = create_app(loader, api_key=token, output=output / "http")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 120
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("local inference server did not start")
            time.sleep(.05)
        with OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=token, max_retries=0, timeout=120) as client:
            assert client.models.list().data[0].id == "keyprint"
            params = dict(model="keyprint", messages=[{"role":"user", "content":"Say hello in one short sentence."}],
                          max_completion_tokens=64, extra_headers={"Idempotency-Key":"real-inference-client"})
            result = client.chat.completions.create(**params)
            replay = client.chat.completions.create(**params)
            assert result.model_dump() == replay.model_dump()
            assert result.choices[0].message.content.strip()
            assert len(list((output / "http").glob("chatcmpl-*/report.json"))) == 1
            rejections = {}
            for name, override in (("streaming", {"stream":True}), ("system_message", {"messages":[{"role":"system", "content":"test"}]})):
                try:
                    client.chat.completions.create(**{**params, **override})
                except APIStatusError as exc:
                    rejections[name] = exc.status_code
                else:
                    raise AssertionError("unsupported mode accepted: " + name)
            assert all(code == 400 for code in rejections.values())
        with Anthropic(base_url=f"http://127.0.0.1:{port}", api_key=token, max_retries=0, timeout=120,
                       _strict_response_validation=True) as client:
            messages_params = dict(model="keyprint", max_tokens=64,
                messages=[{"role":"user", "content":[{"type":"text", "text":"Say hello in one short sentence."}]}],
                extra_headers={"Idempotency-Key":"real-messages-client"})
            message = client.messages.create(**messages_params)
            assert client.messages.create(**messages_params).model_dump() == message.model_dump()
            assert message.content[0].type == "text" and message.content[0].text.strip()
            assert message.stop_reason in ("end_turn", "max_tokens")
            assert message._request_id == message.id
            try:
                client.messages.create(**{**messages_params, "stream": True})
            except AnthropicStatusError as exc:
                assert exc.status_code == 400 and exc.response.json()["type"] == "error"
            else:
                raise AssertionError("unsupported Messages streaming accepted")
        assert len(list((output / "http").glob("chatcmpl-*/report.json"))) == 1
        assert len(list((output / "http").glob("msg_*/report.json"))) == 1
        saved = json.loads(next((output / "http").glob("chatcmpl-*/report.json")).read_text())
        saved = saved.get("report", saved)
        saved_text = saved.get("text", saved.get("rendered_carriers", {}).get("visible_text"))
        assert saved_text == result.choices[0].message.content
        saved_message = json.loads((output / "http" / message.id / "report.json").read_text())
        saved_message = saved_message.get("report", saved_message)
        message_text = saved_message.get("text", saved_message.get("rendered_carriers", {}).get("visible_text"))
        assert message_text == message.content[0].text
        assert saved_message["usage"]["prompt_tokens"] == message.usage.input_tokens
        assert saved_message["usage"]["completion_tokens"] == message.usage.output_tokens
        return {"openai_http":"pass", "text":saved_text, "finish_reason":result.choices[0].finish_reason,
                "usage":result.usage.model_dump(), "idempotency_exact_replay":True, "model_attempts":2,
                "unsupported_rejections":rejections,
                "anthropic_messages":{"status":"pass", "text":message.content[0].text,
                    "stop_reason":message.stop_reason, "usage":message.usage.model_dump(),
                    "idempotency_exact_replay":True, "private_receipt_matches":True,
                    "streaming_rejection":400},
                "hosted_provider_calls":False}
    finally:
        server.should_exit = True
        thread.join(timeout=180)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("local inference server failed to shut down")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--http-client", action="store_true")
    args = parser.parse_args()
    if args.backend == "transformers":
        import torch
        torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    key, control = Keyprint.new_key(), Keyprint.new_key()
    for name, value in (("owner.key", key), ("control.key", control)):
        with os.fdopen(os.open(args.output/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), "wb") as stream:
            stream.write(value)
    loader = lambda: (Keyprint.from_mlx if args.backend == "mlx" else Keyprint.from_transformers)(args.model, key=key)
    candidate = loader()
    cases = json.loads(Path(__file__).with_name("inference_cases.json").read_text())
    report = {"backend":args.backend, "keyprint_version":importlib.metadata.version("keyprint"),
              "identity":candidate.identity, "cases":cases, "runs":[], "clients":{},
              "scope":"Actual local inference screen, not a quality, calibrated detector or production acceptance", "engineering_failures":[]}
    for i, case in enumerate(cases):
        order = ("ordinary", "marked") if i % 2 == 0 else ("marked", "ordinary")
        for condition in order:
            row = {"case":case["id"], "condition":condition}
            started = time.perf_counter()
            try:
                result = candidate.generate(case["prompt"], condition=condition, max_tokens=case["max_tokens"],
                                            output=args.output/(case["id"]+"-"+condition))
                completion = result.report.get("payload", result.report)["completion"]
                row.update(text=result.text, completion=completion, usage=result.report["usage"],
                           screens=screens(case, result.text, completion), diagnostic=inspect(candidate, result.text, control))
                assert result.text.strip(), "empty generated text"
                assert 0 < result.report["usage"]["completion_tokens"] <= case["max_tokens"]
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                report["engineering_failures"].append(case["id"]+"-"+condition)
            row["seconds"] = time.perf_counter() - started
            report["runs"].append(row)
            write_report(args.output, report)
            print(case["id"], condition, row.get("completion", row.get("error_type")), flush=True)
    report["comparisons"] = []
    for case in cases:
        rows = {r["condition"]:r for r in report["runs"] if r["case"] == case["id"]}
        if all("text" in r for r in rows.values()):
            a, b = rows["ordinary"]["text"], rows["marked"]["text"]
            report["comparisons"].append({"case":case["id"], "same_text":a == b,
                "word_sequence_similarity":difflib.SequenceMatcher(None, a.split(), b.split()).ratio(),
                "interpretation":"Descriptive wording comparison of independent samples, not meaning preservation"})
    # Actual SDK response objects, explicitly constructed from synthetic prose.
    from openai.types.chat import ChatCompletion
    from anthropic.types import Message
    source = "Hi Maya, please review the draft by Friday at 09:30. Do not publish it before I approve it."
    objects = {
        "openai":ChatCompletion.model_validate({"id":"fixture", "object":"chat.completion", "created":0, "model":"synthetic", "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":source}}]}),
        "anthropic":Message.model_validate({"id":"fixture", "type":"message", "role":"assistant", "model":"synthetic", "stop_reason":"end_turn", "stop_sequence":None,"content":[{"type":"text","text":source}],"usage":{"input_tokens":1,"output_tokens":1}})}
    for provider, response in objects.items():
        try:
            result = getattr(candidate, "rewrite_"+provider)(response, max_tokens=192, output=args.output/(provider+"-rewrite"))
            report["clients"][provider+"_object_local_rewrite"] = {"original":result.original, "text":result.text,
                "status":result.status, "checks":result.checks, "source":"constructed SDK object; actual local model rewrite; no hosted call"}
        except Exception as exc:
            report["clients"][provider+"_object_local_rewrite"] = {"error_type":type(exc).__name__}
            report["engineering_failures"].append(provider+"-rewrite")
        write_report(args.output, report)
    if args.http_client:
        try:
            report["clients"]["local_http"] = client_check(loader, args.output)
        except Exception as exc:
            report["clients"]["local_http"] = {"error_type":type(exc).__name__}
            report["engineering_failures"].append("local_http")
    write_report(args.output, report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write("## Actual inference results\n\n" +
                f"Generated {len(report['runs'])} ordinary/marked outputs. " +
                f"Engineering failures: {len(report['engineering_failures'])}. " +
                f"Outputs flagged by mechanical screens: {len(report['screening_failures'])}.\n\n" +
                "Download the comparison artifact to read both texts. This job gates runtime and protocol contracts, " +
                "not semantic quality or detector acceptance. Provider object rewrites use real local inference; " +
                "they make no hosted GPT/Claude calls. Both local client endpoints support only the documented text subset.\n")
    return int(bool(report["engineering_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
