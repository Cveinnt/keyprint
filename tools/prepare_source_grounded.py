"""Freeze a source-grounded development sample without output-based selection.

Only source/instruction/category and fixed length bounds determine selection.
Existing dataset responses are never used for selection, prompts or scoring.
Dataset content remains outside the MIT SDK; this writes private inputs plus a
public provenance manifest, not a redistribution of source articles.
"""
import argparse
import hashlib
import json
from pathlib import Path

from multikey_prose_pilot import digest, save

REPO = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SALT = "keyprint-source-grounded-v1-2026-09-28"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select(rows, count=16):
    if type(count) is not int or count < 1:
        raise ValueError("positive sample count required")
    unique = {}
    eligible = 0
    for index, row in enumerate(rows):
        if row.get("category") != "summarization":
            continue
        context, instruction = row.get("context"), row.get("instruction")
        if (not isinstance(context, str) or not isinstance(instruction, str)
                or not instruction.strip() or not 1200 <= len(context) <= 6000):
            continue
        eligible += 1
        source_hash, instruction_hash = sha(context), sha(instruction)
        item = {"row_index": index, "source_sha256": source_hash, "instruction_sha256": instruction_hash,
                "selection_sha256": sha(SALT + "\0" + source_hash), "source_chars": len(context),
                "source": context, "instruction": instruction}
        # Same source can appear under several tasks. Choose its smallest
        # instruction hash independently of file ordering or answer contents.
        if source_hash not in unique or instruction_hash < unique[source_hash]["instruction_sha256"]:
            unique[source_hash] = item
    ordered = sorted(unique.values(), key=lambda row: (row["selection_sha256"], row["source_sha256"]))
    if len(ordered) < count:
        raise ValueError("Insufficient eligible unique sources")
    return ordered[:count], {"eligible_records": eligible, "eligible_unique_sources": len(unique)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("Use the declared dataset snapshot")
    data = args.snapshot / "databricks-dolly-15k.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    selected, counts = select(rows)
    args.output.mkdir(mode=0o700)
    public, private = args.output / "public", args.output / "private"
    public.mkdir(); private.mkdir(mode=0o700)
    instruction = ("Complete the instruction using only the source passage. Preserve its facts and uncertainty. "
                   "Do not add outside facts, explanations or promises. Answer in English. "
                   "Use the requested format; when no length is specified, write a concise summary.\n\n")
    cases = []
    for row in selected:
        cases.append({**row, "id": row["source_sha256"][:16],
                      "prompt": instruction + "Instruction:\n" + row["instruction"] + "\n\nSource:\n" + row["source"]})
    save(private / "cases.json", cases)
    plan = {"schema": "keyprint.source-grounded-selection.v1", "dataset": REPO, "revision": REVISION,
            "url": "https://huggingface.co/datasets/" + REPO + "/tree/" + REVISION,
            "dataset_sha256": digest(data), "card_sha256": digest(args.snapshot / "README.md"),
            "selector_sha256": digest(Path(__file__)), "private_cases_sha256": digest(private / "cases.json"),
            "records": len(rows), **counts, "selected": len(cases), "salt": SALT,
            "selection": "Category summarization; source 1200-6000 Unicode characters; nonempty instruction; deduplicate exact source by smallest instruction hash; take first 16 sorted fixed salted source hashes. No output/response/reference-answer/quality filtering.",
            "cases": [{k: v for k, v in r.items() if k not in ("source", "instruction", "prompt")} for r in cases],
            "intended_execution": "Unchanged pinned Qwen3.5 experimental-wide, temperature .7, top_k 100, max_tokens 768, all four existing keys, ordinary/marked pairs. Freeze complete inference plan and source before execution; no retries or rewriting.",
            "review": "Rate generated claims against the supplied source, important task coverage, requested format and language before unblinding condition/key/counts. Original dataset responses are not treated as gold truth. Retain every error and uncertain judgment.",
            "scope": "Fresh to this development evaluation, not known held-out from model training. English summarization only. Additional natural-task evidence; does not replace failed multilingual expansion stress tests, registered acceptance criteria or detector calibration.",
            "license": "CC-BY-SA-3.0 dataset per pinned Databricks card; attribution: Databricks databricks-dolly-15k and its listed contributors/source material. Source articles and derived inputs are separate from the MIT SDK, retained privately here; no source-data republication in this manifest.",
            "generation_started": False, "quality_acceptance": False, "launch_ready": False}
    save(public / "selection.json", plan)
    print(json.dumps({k: plan[k] for k in ("records", "eligible_records", "eligible_unique_sources", "selected", "revision")}, indent=2))


if __name__ == "__main__":
    main()
