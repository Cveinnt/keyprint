"""Real pinned Qwen JSON requests, typed local clients and grammar replay.

Retain every attempt. These format and integration checks do not qualify
semantics, detector power, serving cost or production readiness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from keyprint import Keyprint
from keyprint.structured import JsonConstraint
from validate_structured import Record, clients
from validate_compatibility import screens, write_report


def audit(candidate, output):
    total = masks = requests = 0
    for path in sorted(output.rglob("report.json")):
        report = json.loads(path.read_text())["report"]
        assert report["kind"] == "generation_trace"
        assert report["target_identity"]["runtime_profile_sha256"] == candidate.identity["runtime_profile_sha256"]
        events, previous = [], "0" * 64
        for sequence, line in enumerate(path.with_name("journal.jsonl").read_bytes().splitlines(keepends=True)):
            row = json.loads(line)
            assert row["sequence"] == sequence and row["previous_sha256"] == previous
            previous = hashlib.sha256(line).hexdigest()
            events.append(row["event"])
        payload = report["payload"]
        tokens = payload["committed_token_ids"]
        assert report["usage"]["completion_tokens"] == len(tokens)
        assert events[-1]["outcome"] == payload["completion"]
        if "structured_output" in report:
            constraint = JsonConstraint.for_mlx(report["structured_output"]["identity"]["schema"], candidate._backend.tokenizer)
            assert events[0]["constraint"] == constraint.identity == report["structured_output"]["identity"]
            allowed = None
            committed = []
            for event in events:
                if event["kind"] == "grammar_mask":
                    assert allowed is None
                    allowed = constraint.allowed()
                    assert event["allowed_sha256"] == hashlib.sha256(allowed.tobytes()).hexdigest()
                    assert event["allowed_count"] == int(allowed.sum())
                    masks += 1
                elif event["kind"] == "committed_step":
                    token = event["constraint_token_id"]
                    assert allowed is not None and allowed[token]
                    constraint.commit(token)
                    committed.append(token)
                    allowed = None
            assert committed == tokens and allowed is None
            text = report["rendered_carriers"]["visible_text"]
            assert candidate._backend.tokenizer.decode(tokens, skip_special_tokens=True) == text
            result = constraint.finish(text, payload["completion"])
            assert all(report["structured_output"][k] == v for k, v in result.items())
        total += len(tokens)
        requests += 1
    return {"status": "pass", "requests": requests, "tokens": total, "replayed_masks": masks,
            "scope": "Journal chain, mask replay, commits, decoder and independent JSON validation; not model-logit replay or semantic acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=("reference", "experimental-fast", "experimental-native"), default="reference")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(key)
    base = next(c for c in json.loads(Path(__file__).with_name("inference_cases.json").read_text()) if c["id"] == "json")
    cases = [{**base, "id": "unconstrained-json"}, {**base, "json_schema": Record.model_json_schema()},
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
    report = {"backend": f"Pinned Qwen3-8B / MLX {args.execution} / JSON grammar", "cases": cases,
        "runs": [], "clients": {}, "engineering_failures": [], "status": "started", "scope": __doc__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_report(args.output, report)
    try:
        loader = lambda: Keyprint.from_mlx(args.model, key=key, execution=args.execution)
        candidate = loader()
        report["identity"] = candidate.identity
        for i, case in enumerate(cases):
            for condition in (("ordinary", "marked") if i % 2 == 0 else ("marked", "ordinary")):
                result = candidate.generate(case["prompt"], condition=condition, max_tokens=case["max_tokens"],
                    output=args.output / (case["id"] + "-" + condition), json_schema=case.get("json_schema"))
                completion = result.report["payload"]["completion"]
                report["runs"].append({"case": case["id"], "condition": condition, "text": result.text,
                    "completion": completion, "usage": result.report["usage"],
                    "structured_output": result.report.get("structured_output"), "screens": screens(case, result.text, completion)})
                write_report(args.output, report)
                print(case["id"], condition, completion, flush=True)
        capped = candidate.generate(base["prompt"], max_tokens=1, json_schema=Record.model_json_schema(), output=args.output / "token-cap")
        assert capped.report["structured_output"]["status"] == "incomplete"
        report["token_cap"] = {"text": capped.text, "structured_output": capped.report["structured_output"]}
        report["clients"] = clients(candidate, args.output, base["prompt"], loader=loader)
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
