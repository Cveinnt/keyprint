"""Reconcile complete-caller parity receipts with the installed experimental source."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_mlx_capacity import read_journal
from benchmark_serving import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root, public = args.run, args.run / "public"
    destination = public / "integrity.json"
    if destination.exists(): raise ValueError("Never overwrite an existing audit")
    plan = json.loads((public / "plan.json").read_text())
    summary = json.loads((public / "summary.json").read_text())
    from keyprint.backends.mlx import ASSETS
    from keyprint.experimental.fast_mlx import execution_specification
    label = plan.get("candidate_label", "fast")
    if label == "bounded_reference":
        from keyprint.backends.mlx_bounded import execution_specification
    elif label == "native":
        from keyprint.experimental.native_mlx import execution_specification
    elif label != "fast":
        raise ValueError("Unknown measured execution")
    from transformers import AutoTokenizer
    if sha(Path(__file__).with_name("validate_fast_caller.py")) != plan["script_sha256"]:
        raise ValueError("Original caller validator changed")
    if sha(Path(__file__).with_name("inference_cases.json")) != plan["cases_sha256"]:
        raise ValueError("Original cases changed")
    if plan[label + "_identity"]["specification"]["execution"] != execution_specification():
        raise ValueError("Installed experimental execution differs from measured source")
    model = Path(plan["model_path"])
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha(model / name) != ASSETS[name]: raise ValueError("Pinned tokenizer differs")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    count, pairs, outputs = 0, 0, 0
    for case in plan["cases"]:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                                           tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
        prompt_sha = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        for condition in ("ordinary", "marked"):
            name = f"{case['id']}-{condition}"
            row = json.loads((public / (name + ".json")).read_text())
            if row["status"] != "pass" or row["errors"] or not all(row["checks"].values()):
                raise ValueError("Failed or incomplete pair")
            reports = []
            for execution in ("reference", label):
                path = root / (name + "-" + execution)
                item = row["executions"][execution]
                for filename, field in (("report.json", "report_sha256"), ("journal.jsonl", "journal_sha256")):
                    if sha(path / filename) != item[field]: raise ValueError("Stored artifact hash differs")
                report = json.loads((path / "report.json").read_text())
                events = read_journal(path / "journal.jsonl")
                starts = [e for e in events if e["kind"] == "response_started"]
                terminals = [e for e in events if e["kind"] == "response_terminal"]
                commits = [e for e in events if e["kind"] == "committed_step"]
                if len(starts) != 1 or len(terminals) != 1 or events[-1] != terminals[0]:
                    raise ValueError("Exactly one complete journal lifecycle required")
                start, end, payload = starts[0], terminals[0], report["payload"]
                identity = plan[execution + "_identity"]
                if (start["runtime_profile_sha256"] != identity["runtime_profile_sha256"]
                        or report["target_identity"]["runtime_profile_sha256"] != identity["runtime_profile_sha256"]
                        or start["prompt_sha256"] != prompt_sha or start["prompt_length"] != len(ids)
                        or start["condition"] != condition or payload["assigned_condition"] != condition
                        or start["max_tokens"] != case["max_tokens"]):
                    raise ValueError("Prompt, condition, cap or execution identity differs")
                if (len(commits) != len(payload["committed_token_ids"]) or len(commits) != row["tokens_per_execution"]
                        or len(commits) != end["sampled_tokens"] or report["kind"] != "generation_trace"
                        or [e["index"] for e in commits] != list(range(len(commits)))
                        or [r["token_id"] for r in payload["sampling_records"]] != payload["committed_token_ids"]
                        or end["error_type"] is not None or end["journal_broken"] is not False
                        or report["rendered_carriers"]["visible_text"] != item["text"]):
                    raise ValueError("Token count, terminal or text differs")
                count += len(commits)
                outputs += 1
                reports.append(report)
            for field in ("sampling_records", "committed_token_ids", "literal_diagnostics", "completion"):
                if reports[0]["payload"][field] != reports[1]["payload"][field]:
                    raise ValueError(f"Pair differs: {field}")
            if reports[0]["rendered_carriers"] != reports[1]["rendered_carriers"]:
                raise ValueError("Rendered pair differs")
            pairs += 1
    if (summary["status"] != "pass" or summary["pairs"] != pairs or summary["outputs"] != outputs
            or summary["tokens_per_execution"] * 2 != count):
        raise ValueError("Summary differs")
    result = {"status": "pass", "pairs": pairs, "outputs": outputs, "verified_committed_tokens": count,
              "plan_sha256": sha(public / "plan.json"), "summary_sha256": sha(public / "summary.json"),
              "auditor_sha256": sha(Path(__file__)),
              "scope": "Stored artifacts, prompts, identity and execution parity; no semantic, detection or serving acceptance"}
    with destination.open("x") as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
