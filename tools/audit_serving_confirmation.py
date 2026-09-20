"""Reconcile frozen-runtime confirmation artifacts and retain every Dolly pair."""
import argparse
import json
import math
from pathlib import Path

from analyze_mlx_capacity import read_journal
from audit_serving import verify_output
from audit_capped_rendering import verify_rendering
from benchmark_serving import sha, summarize
from serving_confirmation import select_cases, CORPUS_SHA
from validate_compatibility import screens, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    root, public = args.run, args.run / "public"
    destination = public / "integrity.json"
    if destination.exists():
        raise ValueError("Never overwrite an existing serving audit")
    plan = json.loads((public / "plan.json").read_text())
    summary = json.loads((public / "summary.json").read_text())
    load = json.loads((public / "load.json").read_text())
    if (sha(Path(__file__).with_name("serving_confirmation.py")) != plan["script_sha256"]
            or sha(Path(__file__).with_name("benchmark_serving.py")) != plan["shared_benchmark_sha256"]):
        raise ValueError("Frozen confirmation or shared analysis changed")
    cases = select_cases(args.corpus)
    if plan["cases"] != cases or plan["corpus_sha256"] != CORPUS_SHA:
        raise ValueError("Selected original corpus workload differs")
    if load["identity"]["runtime_profile_sha256"] != plan["expected_runtime_sha256"]:
        raise ValueError("Runtime changed after freezing candidate")
    byte_audit = 'capped_utf8.py' in load['identity']['specification']['execution']['sources']
    if byte_audit:
        from keyprint.experimental.fast_reporting import _experimental_target
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        from keyprint._engine.legacy._impl.research.token_channel_host import EOS
        _experimental_target(load['identity'])
        token_bytes = runtime_binding(max_steps=load['identity']['max_steps']).token_bytes
    from keyprint.backends.mlx import ASSETS
    from transformers import AutoTokenizer
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha(args.model / name) != ASSETS[name]:
            raise ValueError("Pinned tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    cases = plan["cases"]
    expected = []
    for case in cases:
        for condition in ("ordinary", "marked"):
            expected.append((case, -1, condition, f"warmup-{case['id']}--1-{condition}", 32))
    for repeat in range(plan["repeats"]):
        for case in cases:
            for condition in ("ordinary", "marked"):
                expected.append((case, repeat, condition, f"{case['id']}-{repeat}-{condition}", case["max_tokens"]))
    rows, comparisons, token_count, capped = [], [], 0, 0
    byte_checks = []
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
        if byte_audit:
            byte_checks.append({'id': name, **verify_rendering(report, events, token_bytes, EOS)})
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
            or summary["attempts"] != 64 or summary["warmups"] != 16 or summary["analysis"] != analysis
            or not math.isclose(independent_ratio, analysis["seconds_per_committed_token"]["geometric_mean_ratio"], rel_tol=1e-12)
            or summary["production_overhead_accepted"] is not False):
        raise ValueError("Timing summary differs or overclaims acceptance")
    comparison = {"backend": "Pinned experimental Qwen3-8B / MLX; eight declared Dolly tasks, four retained pairs each; Databricks Dolly, CC-BY-SA-3.0",
                  "cases": [{**case, "id": f"{case['id']}-{repeat}"} for repeat in range(plan["repeats"]) for case in cases],
                  "runs": comparisons}
    comparison["source"] = {k:plan[k] for k in ("source_url", "attribution", "corpus_license", "corpus_revision", "corpus_sha256", "selection")}
    write_report(root, comparison)
    page = public / "comparison.html"
    page.write_text(page.read_text()+'<footer><p>Instruction and context material: <a href="https://huggingface.co/datasets/databricks/databricks-dolly-15k">Databricks Dolly</a>, <a href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a>. Adaptation: original instruction plus a Context label. Outputs are newly generated; this study does not approve their correctness.</p></footer>')
    result = {"status": "pass", "measured_outputs": len(rows), "warmups": 16,
              "verified_committed_tokens_including_warmups": token_count, "measured_capped_outputs": capped,
              "mechanical_flags": len(comparison["screening_failures"]),
              "independent_primary_ratio": independent_ratio, "analysis": analysis,
              "plan_sha256": sha(public / "plan.json"), "summary_sha256": sha(public / "summary.json"),
              "script_sha256": sha(Path(__file__)),
              "byte_audit": {'required': byte_audit, 'outputs': byte_checks,
                             'script_sha256': sha(Path(__file__).with_name('audit_capped_rendering.py'))},
              "scope": "Stored artifact and arithmetic integrity only; no model-head replay, semantic-quality or native-server-overhead acceptance"}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
