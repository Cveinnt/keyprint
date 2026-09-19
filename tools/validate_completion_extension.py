"""Extend every token-limited corpus response, preserving its exact sampled prefix.

Research replay, not a production resume API or crash recovery. Original artifacts
stay unchanged. Instance-local observers return the original sampler results and
stop on any prefix mismatch before new randomness is allowed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

import numpy as np
from keyprint import Keyprint
from analyze_mlx_capacity import read_journal
from validate_null_corpus import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_identity(candidate, target):
    # Public reports contain the identity digest/projection, not the full private
    # specification. Verify the digest and all projected runtime fields.
    specification = json.dumps(candidate.get("specification"), sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()
    return (target.get("specification_digest_verified") is True
            and hashlib.sha256(specification).hexdigest() == candidate.get("runtime_profile_sha256")
            and all(candidate.get(k) == target.get(k) for k in (
                "version", "runtime_profile_sha256", "score_namespace_sha256", "max_steps", "deployment_calibrated")))


class PrefixReplay:
    def __init__(self, events, tokens, records, fresh_bits=secrets.randbits):
        self.heads = [e for e in events if e["kind"] == "prepared_step"]
        self.draws = [e for e in events if e["kind"] == "random_bits_returned"]
        self.tokens, self.records = tokens, records
        self.step, self.replayed_draws, self.new_draws = 0, 0, 0
        self.fresh_bits = fresh_bits
        self.prepared = False
        if not tokens or len(self.heads) != len(tokens) or len(records) != len(tokens):
            raise ValueError("Complete saved token/probability/head prefix required")

    def prepare(self, logits):
        if self.prepared:
            raise ValueError("Previous step not checked")
        if self.step < len(self.tokens):
            expected = self.heads[self.step]
            if expected["index"] != self.step or hashlib.sha256(logits.tobytes()).hexdigest() != expected["raw_logits_sha256"]:
                raise ValueError("Saved raw logits differ")
        self.prepared = True

    def bits(self, count):
        if not self.prepared:
            raise ValueError("Randomness requested before checked preparation")
        if self.step < len(self.tokens):
            if self.replayed_draws >= len(self.draws):
                raise ValueError("Saved randomness exhausted before prefix end")
            draw = self.draws[self.replayed_draws]
            if draw["bits"] != count or draw["sample_index"] != self.step:
                raise ValueError("Saved random request differs")
            self.replayed_draws += 1
            return int(draw["value_decimal"])
        if self.replayed_draws != len(self.draws):
            raise ValueError("Unconsumed saved randomness at extension boundary")
        self.new_draws += 1
        return self.fresh_bits(count)

    def commit(self, token, record):
        if not self.prepared:
            raise ValueError("Unprepared commit")
        if self.step < len(self.tokens):
            if token != self.tokens[self.step] or record != self.records[self.step]:
                raise ValueError("Saved token/probability/random transcript differs")
        self.step += 1
        self.prepared = False


def extend(candidate, prompt, saved, events, *, output):
    payload = saved["payload"]
    replay = PrefixReplay(events, payload["committed_token_ids"], payload["sampling_records"])
    start = next(e for e in events if e["kind"] == "response_started")
    encoded = candidate._backend.encode_prompt(prompt)
    encoded_hash = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if encoded_hash != start["prompt_sha256"] or not same_identity(candidate.identity, saved["target_identity"]):
        raise ValueError("Original prompt or sampler identity differs")
    if payload["completion"] != "length" or start["max_tokens"] != 512 or len(replay.tokens) != 512:
        raise ValueError("Only the original 512-token capped paths may be extended")
    facade = candidate._candidate
    original_factory = facade._core.pipeline
    original_run = facade.run_response

    def observed_factory(*args, **kwargs):
        pipeline = original_factory(*args, **kwargs)
        original_step = pipeline.step

        def observed_step(logits, random_bits):
            replay.prepare(logits)
            result = original_step(logits, random_bits)
            replay.commit(result.token_id, pipeline._raw.sampling_records[-1])
            return result

        pipeline.step = observed_step
        return pipeline

    def observed_run(model, ids, **settings):
        settings["random_bits"] = replay.bits
        return original_run(model, ids, **settings)

    facade._core.pipeline = observed_factory
    facade.run_response = observed_run
    try:
        result = candidate.generate(prompt, max_tokens=1024, condition=start["condition"], output=output)
        if replay.step < 512 or replay.replayed_draws != len(replay.draws):
            raise ValueError("Original prefix did not finish replaying")
        return result, {"verified_prefix_tokens": 512, "replayed_draws": replay.replayed_draws,
                        "new_draws": replay.new_draws, "all_prefix_logits_probabilities_tokens_draws_match": True}
    finally:
        # PublicCandidate removes its factory observer itself; clear our observer
        # too if an earlier input/factory failure occurred. Never patch a class.
        facade._core.__dict__.pop("pipeline", None)
        facade.__dict__.pop("run_response", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_plan = json.loads((args.study / "public/plan.json").read_text())
    originals = [json.loads(p.read_text()) for p in sorted((args.study / "public").glob("power-*.json"))]
    if len(originals) != 24 or any(r.get("error") for r in originals):
        raise ValueError("Complete original 24-output study required")
    selected = [r for r in originals if r["completion"] == "length"]
    if not selected:
        raise ValueError("No capped responses in source study")
    keys = [(args.study / f"owner-{i}.key").read_bytes() for i in range(2)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != source_plan["key_commitments"]:
        raise ValueError("Source key commitments differ")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    for i, key in enumerate(keys):
        with os.fdopen(os.open(args.output / f"owner-{i}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(key)
    declaration = {"scope": "Exact-prefix completion extension; not fresh independent inference, semantic quality or serving-latency acceptance",
        "script_sha256": sha(__file__), "journal_reader_sha256": sha(Path(__file__).with_name('analyze_mlx_capacity.py')),
        "source_plan_sha256": sha(args.study / "public/plan.json"), "max_tokens_before": 512, "max_tokens_after": 1024,
        "selection": "Every original response ending at token limit, no score/text quality selection",
        "cases": [r["id"] for r in selected], "key_commitments": source_plan["key_commitments"],
        "source_reports": {r["id"]: sha(args.study / r["id"] / "report.json") for r in selected},
        "source_journals": {r["id"]: sha(args.study / r["id"] / "journal.jsonl") for r in selected},
        "source_public_rows": {r["id"]: sha(args.study / "public" / (r["id"] + '.json')) for r in selected},
        "rule": "Abort on mismatched original prefix before new randomness; retain all attempts; no retries"}
    write(public / "plan.json", declaration)
    rows, failure = [], None
    try:
        candidates = [Keyprint.from_mlx(args.model, key=key) for key in keys]
        for source in selected:
            name = source["id"]
            row = {"id": name, "condition": source["condition"], "key_index": source["key_index"],
                   "original_text": source["text"], "original_words": source["words"]}
            started = time.monotonic()
            try:
                saved = json.loads((args.study / name / "report.json").read_text())["report"]
                events = read_journal(args.study / name / "journal.jsonl")
                index = int(name.split('-')[1])
                result, audit = extend(candidates[source["key_index"]], source_plan["tasks"][index]["prompt"],
                                       saved, events, output=args.output / name)
                row.update(text=result.text, completion=result.report["payload"]["completion"],
                           words=len(result.text.split()), usage=result.report["usage"], **audit,
                           rendered_text_starts_with_original=result.text.startswith(source["text"]))
            except Exception as exc:
                row["error"] = type(exc).__name__
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            write(public / (name + '.json'), row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('text', 'original_text')}), flush=True)
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        complete = len(rows) == len(selected) and not failure and not any(r.get('error') for r in rows)
        write(public / "summary.json", {"status": "completed" if complete else "incomplete", "failure": failure,
            "planned": len(selected), "attempts": len(rows), "errors": sum(bool(r.get('error')) for r in rows),
            "ended_at_eos": sum(r.get('completion') == 'eos' for r in rows),
            "still_capped": sum(r.get('completion') == 'length' for r in rows),
            "original_uncapped_responses_unchanged": 24 - len(selected), "scope": declaration['scope']})
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
