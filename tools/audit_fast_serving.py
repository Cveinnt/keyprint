"""Audit all interleaved timing artifacts and publish every generated pair.

Verifies stored paths and arithmetic, not model-head replay or semantic quality.
"""
import argparse
import json
import math
from pathlib import Path

from analyze_mlx_capacity import read_journal
from audit_serving import verify_output
from benchmark_fast_serving import CELLS, REPEATS, analyze, cell_order
from benchmark_serving import sha
from validate_compatibility import screens, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root, public = args.run, args.run / "public"
    destination = public / "integrity.json"
    if destination.exists():
        raise ValueError("Never overwrite an existing audit")
    plan = json.loads((public / "plan.json").read_text())
    summary = json.loads((public / "summary.json").read_text())
    load = json.loads((public / "load.json").read_text())
    for filename, field in (("benchmark_fast_serving.py", "script_sha256"),
                            ("benchmark_serving.py", "shared_benchmark_sha256"),
                            ("inference_cases.json", "cases_sha256")):
        if sha(Path(__file__).with_name(filename)) != plan[field]:
            raise ValueError("Declared benchmark or cases changed")
    if (plan["repeats"] != REPEATS or plan["measured_requests"] != 96
            or plan["cells"] != [list(c) for c in CELLS]):
        raise ValueError("Study design differs")
    from keyprint.backends.mlx import ASSETS
    from transformers import AutoTokenizer
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha(args.model / name) != ASSETS[name]:
            raise ValueError("Pinned tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    expected = [(plan["cases"][0], -1, e, c, True) for e, c in CELLS]
    for repeat in range(REPEATS):
        for index, case in enumerate(plan["cases"]):
            expected.extend((case, repeat, e, c, False) for e, c in cell_order(repeat, index))
    rows, comparisons, token_count, capped = [], [], 0, 0
    for case, repeat, execution, condition, warmup in expected:
        name = f"{execution}-{'warmup' if warmup else case['id']}-{repeat}-{condition}"
        row = json.loads((public / (name + ".json")).read_text())
        if (row["id"] != name or row["case"] != case["id"] or row["repeat"] != repeat
                or row["condition"] != condition or row["execution"] != execution
                or row["max_tokens"] != (32 if warmup else case["max_tokens"]) or "error" in row):
            raise ValueError("Missing, failed or substituted output")
        for filename, field in (("report.json", "report_sha256"), ("journal.jsonl", "journal_sha256")):
            if sha(root / name / filename) != row[field]:
                raise ValueError("Generated artifact hash differs")
        report = json.loads((root / name / "report.json").read_text())["report"]
        events = read_journal(root / name / "journal.jsonl")
        ids = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                                           tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
        token_count += verify_output(row, report, events, ids, load["identity"][execution])
        if warmup: continue
        rows.append(row)
        capped += row["completion"] == "length"
        comparisons.append({**row, "case": f"{execution}-{case['id']}-{repeat}",
                            "screens": screens(case, row["text"], row["completion"])})
    result = analyze(rows, [case["id"] for case in plan["cases"]])
    if (summary["status"] != "completed" or summary["failure"] is not None
            or summary["attempts"] != 96 or summary["warmups"] != 4
            or summary["analysis"] != result or summary["production_overhead_accepted"] is not False):
        raise ValueError("Summary differs or overclaims acceptance")
    lookup = {(r["execution"], r["case"], r["repeat"], r["condition"]): r for r in rows}
    independent = {}
    for condition in ("ordinary", "marked"):
        ratios = []
        for case in plan["cases"]:
            for repeat in range(REPEATS):
                fast, reference = (lookup[e, case["id"], repeat, condition] for e in ("fast", "reference"))
                ratios.append(math.log((fast["seconds"] / fast["completion_tokens"]) /
                                       (reference["seconds"] / reference["completion_tokens"])))
        independent[condition] = math.exp(math.fsum(ratios) / len(ratios))
        if not math.isclose(independent[condition], result["fast_over_reference"][condition]["geometric_mean_seconds_per_token_ratio"], rel_tol=1e-12):
            raise ValueError("Independent timing ratio differs")
    comparison = {"backend": "Pinned Qwen3-8B / MLX; interleaved reference and experimental execution",
                  "cases": [{**case, "id": f"{execution}-{case['id']}-{repeat}"}
                            for execution in ("reference", "fast") for repeat in range(REPEATS) for case in plan["cases"]],
                  "runs": comparisons}
    write_report(root, comparison)
    audit = {"status": "pass", "measured_outputs": len(rows), "warmups": 4,
             "verified_committed_tokens_including_warmups": token_count, "measured_capped_outputs": capped,
             "mechanical_flags": len(comparison["screening_failures"]),
             "independent_fast_over_reference": independent, "analysis": result,
             "plan_sha256": sha(public / "plan.json"), "summary_sha256": sha(public / "summary.json"),
             "audit_sha256": sha(Path(__file__)),
             "scope": "Stored receipts and timing arithmetic only; no model-head replay, semantic quality or native-server acceptance"}
    with destination.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
