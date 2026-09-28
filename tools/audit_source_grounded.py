"""Audit all 128 source-grounded receipts without assigning quality or detection passes."""
import argparse
import json
from pathlib import Path
import re

from audit_pydantic_ai import audit_journal
from audit_wide_mlx import audit_steps
from multikey_prose_pilot import digest, save
from validate_source_grounded import schedule


def reconcile(plan, cases, rubrics, rows, review):
    """Reject missing/extra attempts and any mismatch in the blinded review."""
    expected = schedule(cases)
    if plan["schedule"] != expected:
        raise ValueError("Schedule differs from all sixteen cases and four keys")
    signature = lambda r: (r["case"], r["key_slot"], r["condition"])
    if [signature(r) for r in rows] != [signature(r) for r in expected]:
        raise ValueError("Missing, reordered, duplicate or undeclared attempts")
    ids = [r["review_id"] for r in rows]
    if len(set(ids)) != 128 or any(not re.fullmatch(r"[0-9a-f]{12}", uid) for uid in ids):
        raise ValueError("Invalid or duplicate review identity")
    by_case = {r["id"]: r for r in cases}
    by_rubric = {r["id"]: r for r in rubrics}
    if len(rubrics) != 16 or set(by_rubric) != set(by_case):
        raise ValueError("Incomplete source rubrics")
    if len(review) != 128 or {r["review_id"] for r in review} != set(ids):
        raise ValueError("Blinded review is incomplete")
    by_row = {r["review_id"]: r for r in rows}
    for item in review:
        row = by_row[item["review_id"]]
        exact = {"review_id": row["review_id"], "case": by_case[row["case"]],
                 "rubric": by_rubric[row["case"]], "text": row.get("text", ""),
                 "completion": row.get("completion"), "error_type": row.get("error_type")}
        if item != exact:
            raise ValueError("Blinded text, source, rubric or metadata differs")


def audit_output(row, report, events, binding, tokenizer, case, identity, cap):
    ids = report["committed_token_ids"]
    if not ids or len(ids) > cap or any(type(i) is not int or not 0 <= i < len(binding.pieces)
            or (binding.pieces[i] is None and i not in binding.eos_ids) for i in ids):
        raise ValueError("Illegal committed token or cap")
    if (report["kind"] != "generation_trace" or report["identity"] != identity
            or events[0]["identity"] != identity or report["condition"] != row["condition"]
            or events[0]["condition"] != row["condition"] or len(ids) != row["tokens"]
            or row["model_calls"] != len(ids) or report["model_calls"] != len(ids)):
        raise ValueError("Identity, condition or calls differ")
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if events[0]["prompt_token_ids"] != prompt or report["usage"] != {
            "prompt_tokens": len(prompt), "completion_tokens": len(ids), "total_tokens": len(prompt) + len(ids)}:
        raise ValueError("Actual prompt or usage differs")
    carrier = report["carrier_rendering"][0]
    raw = b"".join(binding.pieces[i] or b"" for i in ids)
    pending = bytes.fromhex(carrier["pending_utf8_hex"])
    if (raw != row["text"].encode() + pending or report["text"] != row["text"]
            or (not pending and tokenizer.decode(ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False) != row["text"])):
        raise ValueError("Native bytes, decoder or returned text differ")
    completion = report["completion"]
    if (completion not in {"eos", "length"} or completion != row["completion"]
            or events[-1]["completion"] != completion
            or any(t in binding.eos_ids for t in ids[:-1])
            or (completion == "eos" and (ids[-1] not in binding.eos_ids or pending))
            or (completion == "length" and (len(ids) != cap or ids[-1] in binding.eos_ids))):
        raise ValueError("EOS or cap policy differs")
    return {"review_id": row["review_id"], "tokens": len(ids), "completion": completion,
            "draw_transcript_sha256": audit_steps(events, ids)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path, required=True)
    args = p.parse_args()
    public, private = args.root / "public", args.root / "private"
    read = lambda path: json.loads(path.read_text())
    plan, rows = read(public / "plan.json"), read(private / "runs.json")
    cases, rubrics = read(private / "cases.json"), read(private / "rubrics.json")
    review, identity = read(private / "blind-review.json"), read(public / "identity.json")
    if plan["schema"] != "keyprint.source-grounded-run.v1" or plan["settings"] != {
            "execution": "experimental-wide", "temperature": .7, "top_k": 100, "max_tokens": 768}:
        raise ValueError("Unexpected study profile or settings")
    for name in ("cases.json", "rubrics.json"):
        if digest(private / name) != plan["rubrics_commitment"][name]:
            raise ValueError("Source or rubric commitment differs")
    if digest(private / "cases.json") != plan["selection"]["private_cases_sha256"]:
        raise ValueError("Source selection differs")
    reconcile(plan, cases, rubrics, rows, review)
    import keyprint
    from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding, verify_assets
    from keyprint.integrity import verify
    from mlx_lm.utils import load_tokenizer
    package = Path(keyprint.__file__).parent
    actual_sources = {str(path.relative_to(package)): digest(path) for path in package.rglob("*.py")}
    if actual_sources != plan["sdk_source_sha256"] or verify() != plan["engine_integrity"]:
        raise ValueError("Study SDK sources or frozen engine differ")
    for name, expected in {"validate_source_grounded.py": plan["script_sha256"], **plan["helper_sha256"]}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Registered study source differs")
    if (verify_assets(args.model) != identity["model_assets"]
            or identity["temperature"] != .7 or identity["top_k"] != 100):
        raise ValueError("Model assets or settings differ")
    tokenizer = load_tokenizer(args.model, {"trust_remote_code": False, "local_files_only": True}, eos_token_ids=None)
    if set(tokenizer.eos_token_ids) != {248046}:
        raise ValueError("Runtime EOS differs")
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(), vocabulary_size=248320,
        special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    if binding.digest != identity["tokenizer_binding_sha256"]:
        raise ValueError("Tokenizer binding differs")
    by_case = {r["id"]: r for r in cases}
    audited, failures = [], []
    for row in rows:
        folder = private / row["review_id"]
        if row.get("error_type"):
            # Keep all failed attempts visible. Partial artifacts are hashed,
            # never promoted to successful generation or semantic acceptance.
            failures.append({"review_id": row["review_id"], "error_type": row["error_type"],
                "retained_files": {p.name: digest(p) for p in folder.glob("*") if p.is_file()}})
            continue
        for name, suffix in (("report", ".json"), ("journal", ".jsonl")):
            if digest(folder / (name + suffix)) != row[name + "_sha256"]:
                raise ValueError("Receipt digest differs")
        audited.append(audit_output(row, read(folder / "report.json"), audit_journal(folder / "journal.jsonl"),
            binding, tokenizer, by_case[row["case"]], identity, plan["settings"]["max_tokens"]))
    result = {"schema": "keyprint.source-grounded-receipt-audit.v1", "attempts": len(rows),
              "verified": len(audited), "tokens": sum(r["tokens"] for r in audited),
              "failures": failures, "rows": audited,
              "caps": sum(r["completion"] == "length" for r in audited),
              "distinct_draw_transcripts": len({r["draw_transcript_sha256"] for r in audited}),
              "audit_script_sha256": digest(Path(__file__)),
              "audit_helper_sha256": {n: digest(Path(__file__).with_name(n)) for n in
                  ("audit_wide_mlx.py", "audit_pydantic_ai.py", "validate_source_grounded.py", "multikey_prose_pilot.py")},
              "plan_sha256": digest(public / "plan.json"), "runs_sha256": digest(private / "runs.json"),
              "blind_review_sha256": digest(private / "blind-review.json"),
              "scope": "All planned attempts and blinded source/rubric/text bindings, exact native bytes, prompt, usage, EOS and RNG/commit receipts. Key slots are planned assignments; no independent keyed/model-forward replay, semantic judgment, detector calibration or quality acceptance.",
              "quality_acceptance": False, "detector_calibrated": False, "launch_ready": False}
    save(public / "receipt-audit.json", result)
    print(json.dumps({k: result[k] for k in ("attempts", "verified", "tokens", "failures", "caps")}, indent=2))


if __name__ == "__main__":
    main()
