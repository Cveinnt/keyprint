"""Fail closed on partial, cached or inconsistent SGLang inference receipts."""
import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path

from check_native_receipts import check as check_tokens
from keyprint.experimental.sglang_runtime import REVISION, SOURCE_FILES, VERSIONS
from keyprint.sampling import identity as sampling_identity


def check(root: Path, run_id: str) -> dict:
    if not run_id or (root / "run-id.txt").read_text().strip() != run_id:
        raise ValueError("inference run ID mismatch; cached receipts are not a new run")
    for name in ("exit-code.txt", "contract-exit-code.txt"):
        if (root / name).read_text().strip() != "0":
            raise ValueError(f"{name} is not zero; Docker export success is insufficient")
    report = json.loads((root / "public/comparison.json").read_text())
    cases = json.loads(Path(__file__).with_name("inference_cases.json").read_text())
    if report.get("backend") != "sglang CPU pilot" or report.get("cases") != cases:
        raise ValueError("backend or inference cases differ from the checked source")
    if report.get("engineering_failures") != []:
        raise ValueError("engineering failures present or missing")
    identity = report.get("runtime_identity", {})
    if (identity.get("source_revision") != REVISION or
            identity.get("build_version") not in VERSIONS or
            identity.get("source_files_sha256") != SOURCE_FILES):
        raise ValueError("runtime identity differs from pinned sampling contract")
    expected = {(case["id"], condition): case for case in cases
                for condition in ("ordinary", "marked")}
    rows = report.get("runs", [])
    if len(rows) != len(expected):
        raise ValueError("complete ordinary and marked pairs required for every case")
    seen = set()
    projected = []
    for row in rows:
        pair = (row.get("case"), row.get("condition"))
        if pair not in expected or pair in seen:
            raise ValueError("duplicate or unexpected comparison pair")
        seen.add(pair)
        ids = row.get("token_ids")
        if (not isinstance(ids, list) or not 0 < len(ids) <= expected[pair]["max_tokens"] or
                any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError("invalid returned token IDs")
        if (not isinstance(row.get("text"), str) or not row["text"].strip() or
                row.get("completion") not in {"eos", "length"} or
                row.get("usage", {}).get("completion_tokens") != len(ids)):
            raise ValueError("missing text, completion or accurate token usage")
        projected.append({key: row[key] for key in ("text", "token_ids", "completion")})
    if json.loads((root / "outputs.json").read_text()) != projected:
        raise ValueError("public comparisons differ from retained actual outputs")
    verified = check_tokens(root)
    if report.get("token_path_verification") != verified:
        raise ValueError("reported token verification differs from replayed journals")
    # A matching token path alone cannot prove the output was generated under
    # the condition displayed in the comparison. Bind that label to its trace.
    conditions = {}
    for path in (root / "traces").glob("*.jsonl"):
        events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
        starts = [event for event in events if event["phase"] == "start"]
        if len(starts) != 1 or starts[0].get("condition") not in {"ordinary", "marked"}:
            raise ValueError("one journal start with a known condition required")
        expected_adapter = hashlib.sha256(files("keyprint").joinpath("experimental/native.py").read_bytes()).hexdigest()
        if starts[0].get("adapter_sha256") != expected_adapter:
            raise ValueError("Keyprint adapter source differs from the checked package")
        if starts[0].get("sampling_execution") != sampling_identity():
            raise ValueError("Keyprint sampling execution differs from the checked package")
        ids = tuple(event["token_id"] for event in events if event["phase"] == "selected_tentative")
        conditions[ids] = starts[0]["condition"]
    for row in rows:
        if conditions[tuple(row["token_ids"])] != row["condition"]:
            raise ValueError("displayed condition differs from generation journal")
    return {"status": "pass", "run_id": run_id, "requests": verified["requests"],
            "matching_tokens": verified["matching_tokens"], "runtime_identity": identity,
            "scope": "CPU adapter and returned-token integrity only; not production or quality acceptance"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = check(args.directory, args.run_id)
    (args.directory / "public/integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
