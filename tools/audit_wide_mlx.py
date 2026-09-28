"""Reconcile every planned wide-profile attempt, native bytes and draw journal.

This is receipt/rendering verification, not an independent model-forward replay
or a semantic judgment. It never imports old-profile detection thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_pydantic_ai import audit_journal
from multikey_prose_pilot import digest, save


def audit_draw(draw, returned):
    total = draw["total_weight"]
    if type(total) is not int or total < 1:
        raise ValueError("Invalid integer distribution total")
    bits = (total - 1).bit_length()
    transcript = draw["transcript"]
    if len(transcript) != len(returned) or len(transcript) > 1024:
        raise ValueError("Draw transcript count differs")
    for index, (entry, logged) in enumerate(zip(transcript, returned)):
        point = entry["value"]
        if (type(point) is not int or not 0 <= point < (1 << bits)
                or entry["bit_count"] != bits or logged != {"bits": bits, "value": point}
                or type(entry["accepted"]) is not bool or entry["accepted"] != (point < total)
                or entry["accepted"] != (index == len(transcript) - 1)):
            raise ValueError("Draw rejection transcript differs")
    if ((bits == 0 and (transcript or draw["integer_point"] != 0))
            or (bits != 0 and (not transcript or draw["integer_point"] != transcript[-1]["value"]))):
        raise ValueError("Final integer point differs")


def audit_steps(events, ids):
    if not events or events[0]["phase"] != "start" or events[-1]["phase"] != "complete":
        raise ValueError("Missing terminal journal")
    cursor, draws = 1, []
    for index, token in enumerate(ids):
        if events[cursor] != {"phase": "model_forward", "index": index}:
            raise ValueError("Forward order differs")
        cursor += 1
        prepared = events[cursor]
        if prepared["phase"] != "prepared" or any(len(prepared[k]) != 64 for k in ("raw_logits_sha256", "weights_sha256")):
            raise ValueError("Missing native logit/weight hashes")
        cursor += 1
        returned = []
        while events[cursor]["phase"] == "random_requested":
            requested, received = events[cursor], events[cursor + 1]
            if received["phase"] != "random_returned" or received["bits"] != requested["bits"]:
                raise ValueError("RNG request/return differs")
            returned.append({"bits": received["bits"], "value": received["value"]})
            cursor += 2
        intent, committed = events[cursor], events[cursor + 1]
        if (intent["phase"] != "commit_requested" or intent["token_id"] != token
                or intent["draw"]["token_index"] != token
                or committed != {"phase": "committed", "token_id": token}):
            raise ValueError("Selected token/commit differs")
        audit_draw(intent["draw"], returned)
        draws.extend(returned)
        cursor += 2
    if cursor != len(events) - 1:
        raise ValueError("Extra model/sample/commit events")
    return hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path, required=True)
    args = p.parse_args()
    public, private = args.root / "public", args.root / "private"
    plan = json.loads((public / "plan.json").read_text())
    rows = json.loads((private / "runs.json").read_text())
    identity = json.loads((public / "identity.json").read_text())
    import keyprint
    from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding, verify_assets
    from mlx_lm.utils import load_tokenizer
    package = Path(keyprint.__file__).parent
    for name, expected in plan["sdk_source_sha256"].items():
        if digest(package / name) != expected:
            raise ValueError("Study SDK source differs")
    for name, expected in {"validate_wide_mlx.py": plan["script_sha256"],
                           "long_fidelity_cases.json": plan["cases_sha256"], **plan["helper_sha256"]}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Registered study source differs")
    if verify_assets(args.model) != identity["model_assets"]:
        raise ValueError("Model assets differ")
    tokenizer = load_tokenizer(args.model, {"trust_remote_code": False, "local_files_only": True}, eos_token_ids=None)
    if set(tokenizer.eos_token_ids) != {248046}:
        raise ValueError("Runtime EOS differs")
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(), vocabulary_size=248320,
        special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    if binding.digest != identity["tokenizer_binding_sha256"]:
        raise ValueError("Tokenizer binding differs")
    signature = lambda r: (r["key_slot"], r["case"], r["condition"])
    if [signature(r) for r in rows] != [signature(r) for r in plan["schedule"]]:
        raise ValueError("Missing, reordered, duplicate or undeclared attempts")
    by_case = {c["id"]: c for c in plan["cases"]}
    audited, failures = [], []
    for row in rows:
        folder = private / row["review_id"]
        if row.get("error_type"):
            failures.append({"review_id": row["review_id"], "error_type": row["error_type"]})
            continue
        report = json.loads((folder / "report.json").read_text())
        events = audit_journal(folder / "journal.jsonl")
        for name in ("report", "journal"):
            if digest(folder / (name + (".json" if name == "report" else ".jsonl"))) != row[name + "_sha256"]:
                raise ValueError("Receipt digest differs")
        ids = report["committed_token_ids"]
        if not ids or any(type(i) is not int or not 0 <= i < len(binding.pieces)
                          or (binding.pieces[i] is None and i not in binding.eos_ids) for i in ids):
            raise ValueError("Illegal committed token")
        if (report["identity"] != identity or events[0]["identity"] != identity
                or report["condition"] != row["condition"] or events[0]["condition"] != row["condition"]
                or len(ids) != row["tokens"] or row["model_calls"] != len(ids) or report["model_calls"] != len(ids)):
            raise ValueError("Identity, condition or calls differ")
        expected_prompt = tokenizer.apply_chat_template([{"role": "user", "content": by_case[row["case"]]["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if events[0]["prompt_token_ids"] != expected_prompt or report["usage"]["prompt_tokens"] != len(expected_prompt):
            raise ValueError("Actual prompt differs")
        carrier = report["carrier_rendering"][0]
        raw = b"".join(binding.pieces[i] or b"" for i in ids)
        pending = bytes.fromhex(carrier["pending_utf8_hex"])
        if (raw != row["text"].encode() + pending or report["text"] != row["text"]
                or (not pending and tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["text"])):
            raise ValueError("Native bytes, decoder or returned text differ")
        completion = report["completion"]
        if (completion != row["completion"] or events[-1]["completion"] != completion
                or any(t in binding.eos_ids for t in ids[:-1])
                or (completion == "eos" and (ids[-1] != 248046 or pending))
                or (completion == "length" and len(ids) != plan["settings"]["max_tokens"])):
            raise ValueError("EOS or cap policy differs")
        audited.append({"review_id": row["review_id"], "tokens": len(ids), "completion": completion,
                        "draw_transcript_sha256": audit_steps(events, ids)})
    result = {"schema": "keyprint.wide-mlx-receipt-audit.v1", "attempts": len(rows), "verified": len(audited),
              "tokens": sum(r["tokens"] for r in audited), "failures": failures, "rows": audited,
              "distinct_draw_transcripts": len({r["draw_transcript_sha256"] for r in audited}),
              "audit_script_sha256": digest(Path(__file__)), "plan_sha256": digest(public / "plan.json"),
              "scope": "Receipt, exact native byte rendering, prompt, EOS and RNG/commit reconciliation. No independent forward/weight replay, semantic review or detector calibration.",
              "quality_acceptance": False, "detector_calibrated": False, "launch_ready": False}
    save(public / "receipt-audit.json", result)
    print(json.dumps({k: result[k] for k in ("attempts", "verified", "tokens", "failures", "distinct_draw_transcripts")}, indent=2))


if __name__ == "__main__":
    main()
