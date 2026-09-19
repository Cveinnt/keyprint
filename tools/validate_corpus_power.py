"""Fresh real inference under frozen public-corpus thresholds and independent keys.

Twelve source tasks across six categories, ordinary/marked pairs, no prompt edits
or word-count forcing. All errors/truncations retained; no quality approval.
Only public/ is exportable. Keys and inference journals remain private.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from keyprint import Keyprint
from validate_null_corpus import SOURCE_SHA256, WEIGHTS, sha, write
from compare_score_baselines import score

CATEGORIES = ("general_qa", "open_qa", "brainstorming", "creative_writing", "summarization", "closed_qa")


def select_tasks(source, plan):
    counts = Counter()
    tasks = []
    for item in plan["records"]:
        if item["split"] != "heldout":
            continue
        row = source[item["source_index"]]
        if sha(row["response"].encode()) != item["text_sha256"]:
            raise ValueError("Source response hash differs")
        category = row["category"]
        if category not in CATEGORIES or counts[category] >= 2:
            continue
        if len((row["instruction"] + " " + row["context"]).split()) > 600:
            continue
        prompt = row["instruction"] + ("\n\nReference text:\n" + row["context"] if row["context"] else "")
        tasks.append({"source_index": item["source_index"], "category": category,
                      "prompt": prompt, "prompt_sha256": sha(prompt.encode())})
        counts[category] += 1
    if any(counts[c] != 2 for c in CATEGORIES):
        raise ValueError("Two eligible tasks per declared category required")
    return tasks


def summarize(rows, thresholds):
    panels = {}
    for name, cutoff in thresholds.items():
        groups = {}
        for condition, mode in (("marked", "matching"), ("marked", "other"), ("ordinary", "maximum")):
            attempts = [r for r in rows if r["condition"] == condition]
            values = [r["scores"][mode][name] for r in attempts if not r.get("error")]
            groups[condition + "_" + mode] = {"attempts": len(attempts), "available": len(values),
                "unavailable": len(attempts) - len(values), "above": sum(v > cutoff for v in values)}
        panels[name] = {"threshold": cutoff, "groups": groups}
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--null-study", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if sha(raw) != SOURCE_SHA256:
        raise ValueError("Pinned source checksum differs")
    study_plan = args.null_study / "public/plan.json"
    threshold_file = args.null_study / "public/thresholds.json"
    plan = json.loads(study_plan.read_text())
    thresholds = json.loads(threshold_file.read_text())
    if set(thresholds) != set(WEIGHTS) or not all(np.isfinite(v) for v in thresholds.values()):
        raise ValueError("Both fixed finite thresholds required")
    tasks = select_tasks([json.loads(line) for line in raw.splitlines()], plan)
    keys = [(args.null_study / f"heldout-{i}.key").read_bytes() for i in range(2)]
    if [sha(k) for k in keys] != plan["key_commitments"]["heldout"]:
        raise ValueError("Held-out key commitments differ")
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    for i, key in enumerate(keys):
        with os.fdopen(os.open(args.output / f"owner-{i}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(key)
    declaration = {"scope": "Fresh model-output sensitivity on source tasks; no deployment or semantic-quality qualification",
        "script_sha256": sha(Path(__file__).read_bytes()),
        "score_source_sha256": sha(Path(__file__).with_name('compare_score_baselines.py').read_bytes()),
        "null_source_sha256": sha(Path(__file__).with_name('validate_null_corpus.py').read_bytes()),
        "null_plan_sha256": sha(study_plan.read_bytes()), "threshold_file_sha256": sha(threshold_file.read_bytes()),
        "thresholds": thresholds, "source_sha256": SOURCE_SHA256, "source": plan["source"],
        "attribution": plan["attribution"], "license": plan["license"], "tasks": tasks,
        "selection": "First two eligible held-out source tasks per declared category; prompt plus context <=600 words; no score-based selection",
        "key_commitments": [sha(k) for k in keys], "key_rule": "Task index modulo two; remaining key is wrong-key control",
        "ordering": "Ordinary first on even task index, marked first on odd",
        "temperature": .7, "top_k": 100, "max_tokens": 512,
        "comparison": "Strict greater-than, all attempts retained, no retries or threshold tuning",
        "length_scope": "Null calibration uses human responses of 100-400 words; actual model outputs may fall outside that range and are retained"}
    write(public / "plan.json", declaration)
    rows, failure = [], None
    try:
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        binding = runtime_binding(max_steps=2048)
        candidates = [Keyprint.from_mlx(args.model, key=key) for key in keys]
        if candidates[0].identity != candidates[1].identity:
            raise ValueError("Generation profiles differ across keys")
        write(public / "identity.json", candidates[0].identity)
        for index, task in enumerate(tasks):
            candidate = candidates[index % 2]
            for condition in (("ordinary", "marked") if index % 2 == 0 else ("marked", "ordinary")):
                name = f"power-{index:02d}-{condition}"
                row = {"id": name, "condition": condition, "source_index": task["source_index"],
                       "category": task["category"], "key_index": index % 2}
                started = time.monotonic()
                try:
                    generated = candidate.generate(task["prompt"], condition=condition,
                        max_tokens=512, output=args.output / name)
                    row.update(text=generated.text, completion=generated.report["payload"]["completion"],
                               words=len(generated.text.split()), scores={})
                    row["within_null_word_range"] = 100 <= row["words"] <= 400
                    for label, key in (("matching", keys[index % 2]), ("other", keys[1 - index % 2])):
                        bits = np.asarray([e.bits for e in binding.replay_text(generated.text, key).events if e.eligible])
                        counts = candidate.inspect(generated.text, key=key)
                        if int(bits.sum()) != counts.ones or bits.size != counts.trials:
                            raise ValueError("Literal replay and SDK counts differ")
                        row["scores"][label] = {n: score(bits, w) for n, w in WEIGHTS.items()}
                    row["scores"]["maximum"] = {n: max(row["scores"][k][n] for k in ("matching", "other")) for n in WEIGHTS}
                except Exception as exc:
                    row["error"] = type(exc).__name__
                row["seconds"] = time.monotonic() - started
                rows.append(row)
                write(public / (name + ".json"), row)
                print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        complete = len(rows) == 24 and not failure and not any(r.get("error") for r in rows)
        write(public / "summary.json", {"status": "completed" if complete else "incomplete", "failure": failure,
            "attempts": len(rows), "panels": summarize(rows, thresholds),
            "truncated": sum(r.get("completion") == "length" for r in rows),
            "outside_null_word_range": sum(not r.get("within_null_word_range", False) for r in rows),
            "deployment_calibrated": False, "quality_acceptance": False, "scope": declaration["scope"]})
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
