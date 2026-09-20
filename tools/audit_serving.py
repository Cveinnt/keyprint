"""Reconcile paired timing receipts and expose every generated comparison.

Checks journal chains, prompts, committed tokens, public text and timing
arithmetic. Does not rerun model heads or establish quality or native overhead.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_mlx_capacity import read_journal
from benchmark_serving import sha, summarize
from validate_compatibility import screens, write_report


def verify_output(row, report, events, prompt_ids, identity):
    payload = report["payload"]
    starts = [e for e in events if e["kind"] == "response_started"]
    terminals = [e for e in events if e["kind"] == "response_terminal"]
    if len(starts) != 1 or len(terminals) != 1 or events[-1] != terminals[0]:
        raise ValueError("Exactly one complete journal lifecycle required")
    start, terminal = starts[0], terminals[0]
    packed = json.dumps(prompt_ids, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tokens = payload["committed_token_ids"]
    commits = [e for e in events if e["kind"] == "committed_step"]
    if (start["prompt_sha256"] != hashlib.sha256(packed).hexdigest()
            or start["prompt_length"] != len(prompt_ids)
            or start["condition"] != row["condition"] or start["max_tokens"] != row["max_tokens"]
            or start["runtime_profile_sha256"] != identity["runtime_profile_sha256"]
            or report["target_identity"]["runtime_profile_sha256"] != identity["runtime_profile_sha256"]
            or payload["assigned_condition"] != row["condition"]):
        raise ValueError("Prompt, condition, cap or profile differs")
    if (len(tokens) != row["completion_tokens"] or len(tokens) != len(commits)
            or [r["token_id"] for r in payload["sampling_records"]] != tokens
            or [e["index"] for e in commits] != list(range(len(tokens)))
            or terminal["sampled_tokens"] != len(tokens)
            or report["usage"]["completion_tokens"] != len(tokens)
            or terminal["journal_broken"] is not False or terminal["error_type"] is not None
            or payload["completion"] != row["completion"]):
        raise ValueError("Committed token path or terminal receipt differs")
    if (report["rendered_carriers"]["visible_text"] != row["text"]
            or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]):
        raise ValueError("Public text differs from generated report")
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root, public = args.run, args.run / "public"
    destination = public / "integrity.json"
    if destination.exists():
        raise ValueError("Never overwrite an existing serving audit")
    plan = json.loads((public / "plan.json").read_text())
    summary = json.loads((public / "summary.json").read_text())
    load = json.loads((public / "load.json").read_text())
    if (sha(Path(__file__).with_name("benchmark_serving.py")) != plan["script_sha256"]
            or sha(Path(__file__).with_name("inference_cases.json")) != plan["cases_sha256"]):
        raise ValueError("Frozen benchmark or cases changed")
    from keyprint.backends.mlx import ASSETS
    from transformers import AutoTokenizer
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha(args.model / name) != ASSETS[name]:
            raise ValueError("Pinned tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    cases = plan["cases"]
    expected = []
    for condition in ("ordinary", "marked"):
        expected.append((cases[0], -1, condition, f"warmup--1-{condition}", 32))
    for repeat in range(plan["repeats"]):
        for case in cases:
            for condition in ("ordinary", "marked"):
                expected.append((case, repeat, condition, f"{case['id']}-{repeat}-{condition}", case["max_tokens"]))
    rows, comparisons, token_count, capped = [], [], 0, 0
    for case, repeat, condition, name, cap in expected:
        row = json.loads((public / (name + ".json")).read_text())
        if (row["id"] != name or row["case"] != case["id"] or row["repeat"] != repeat
                or row["condition"] != condition or row["max_tokens"] != cap or "error" in row):
            raise ValueError("Missing, failed or substituted planned output")
        for filename, field in (("report.json", "report_sha256"), ("journal.jsonl", "journal_sha256")):
            if sha(root / name / filename) != row[field]:
                raise ValueError("Generated artifact hash differs")
        report = json.loads((root / name / "report.json").read_text())["report"]
        events = read_journal(root / name / "journal.jsonl")
        ids = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                           return_dict=False)
        token_count += verify_output(row, report, events, ids, load["identity"])
        if repeat < 0:
            continue
        rows.append(row)
        capped += row["completion"] == "length"
        comparisons.append({**row, "case": f"{case['id']}-{repeat}",
                            "screens": screens(case, row["text"], row["completion"])})
    analysis = summarize(rows, [case["id"] for case in cases])
    # Independently recompute the primary point estimate from actual denominators.
    lookup = {(r["case"], r["repeat"], r["condition"]): r for r in rows}
    log_ratios = []
    for repeat in range(plan["repeats"]):
        for case in cases:
            ordinary, marked = (lookup[case["id"], repeat, c] for c in ("ordinary", "marked"))
            log_ratios.append(math.log((marked["seconds"] / marked["completion_tokens"]) /
                                      (ordinary["seconds"] / ordinary["completion_tokens"])))
    independent_ratio = math.exp(math.fsum(log_ratios) / len(log_ratios))
    if (summary["status"] != "completed" or summary["failure"] is not None
            or summary["attempts"] != 48 or summary["warmups"] != 2 or summary["analysis"] != analysis
            or not math.isclose(independent_ratio, analysis["seconds_per_committed_token"]["geometric_mean_ratio"], rel_tol=1e-12)
            or summary["production_overhead_accepted"] is not False):
        raise ValueError("Timing summary differs or overclaims acceptance")
    comparison = {"backend": "Pinned Qwen3-8B / MLX serving study; four retained pairs per prompt",
                  "cases": [{**case, "id": f"{case['id']}-{repeat}"} for repeat in range(plan["repeats"]) for case in cases],
                  "runs": comparisons}
    write_report(root, comparison)
    result = {"status": "pass", "measured_outputs": len(rows), "warmups": 2,
              "verified_committed_tokens_including_warmups": token_count, "measured_capped_outputs": capped,
              "mechanical_flags": len(comparison["screening_failures"]),
              "independent_primary_ratio": independent_ratio, "analysis": analysis,
              "plan_sha256": sha(public / "plan.json"), "summary_sha256": sha(public / "summary.json"),
              "script_sha256": sha(Path(__file__)),
              "scope": "Stored artifact and arithmetic integrity only; no model-head replay, semantic-quality or native-server-overhead acceptance"}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
