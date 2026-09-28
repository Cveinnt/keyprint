"""Retained paired Qwen3.5 study through the explicit experimental SDK path."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

from multikey_prose_pilot import digest, save
from validate_long_fidelity import schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from keyprint import Keyprint
    from keyprint.experimental.wide_mlx import REVISION
    from keyprint.integrity import verify
    import keyprint
    cases_path = Path(__file__).with_name("long_fidelity_cases.json")
    cases = json.loads(cases_path.read_text())
    keys = [(args.prior / f"private/key-{i}").read_bytes() for i in range(4)]
    if len(cases) != 4 or len({c["id"] for c in cases}) != 4 or any(len(k) != 32 for k in keys):
        raise ValueError("Require unchanged four cases and all four existing keys")
    args.output.mkdir(mode=0o700)
    public, private = args.output / "public", args.output / "private"
    public.mkdir(); private.mkdir(mode=0o700)
    package = Path(keyprint.__file__).parent
    plan = {
        "schema": "keyprint.wide-mlx-study.v1", "model_revision": REVISION,
        "cases": cases, "schedule": schedule(cases), "settings": {"temperature": .7, "top_k": 100, "max_tokens": 1024},
        "script_sha256": digest(Path(__file__)), "cases_sha256": digest(cases_path),
        "helper_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("multikey_prose_pilot.py", "validate_long_fidelity.py")},
        "sdk_source_sha256": {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))},
        "engine_integrity": verify(), "prior_plan_sha256": digest(args.prior / "public/plan.json"),
        "key_sha256": [hashlib.sha256(k).hexdigest() for k in keys],
        "key_policy": "All four existing keys, no selection or rotation",
        "failure_policy": "32 attempts, no retries, replacement, rewriting or output normalization. Preserve all errors and caps.",
        "review": "Freeze fact, unsupported-addition, language and prose ratings on shuffled metadata-hidden outputs before condition/key/count review. Assistant development review, not independent human acceptance.",
        "scope": "New pinned Qwen3.5 NFC/wider-head experimental profile on the same four long tasks. No inherited empirical acceptance or detector thresholds. Report matching/next-slot raw counts only, no hit verdicts. Descriptive paired study, not powered noninferiority.",
        "success": "Exact native token bytes, EOS, one draw/commit per forward, source integrity and complete retained attempts. Separately report factual, language and format outcomes for both arms; compatibility success is not quality or launch acceptance.",
    }
    save(public / "plan.json", plan)
    by_case = {c["id"]: c for c in cases}
    rows = []
    for slot, key in enumerate(keys):
        with Keyprint.from_mlx(args.model, key=key, execution="experimental-wide", temperature=.7, top_k=100) as wm:
            identity_path = public / "identity.json"
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != wm.identity:
                    raise ValueError("Binding identity changed between key slots")
            else:
                save(identity_path, wm.identity)
            for attempt in [a for a in plan["schedule"] if a["key_slot"] == slot]:
                row = dict(attempt, review_id=uuid.uuid4().hex[:12])
                started = time.monotonic()
                try:
                    result = wm.generate(by_case[row["case"]]["prompt"], condition=row["condition"],
                        max_tokens=1024, trace=True, output=private / row["review_id"])
                    report = result.report
                    row.update(text=result.text, completion=report["completion"],
                               tokens=len(report["committed_token_ids"]), model_calls=report["model_calls"],
                               report_sha256=digest(result.artifacts / "report.json"),
                               journal_sha256=digest(result.artifacts / "journal.jsonl"))
                    if "".join(t.text for t in result.trace) != result.text:
                        raise ValueError("Token trace differs from returned text")
                    scores = []
                    for other_key in (key, keys[(slot + 1) % 4]):
                        try:
                            inspected = wm.inspect(result.text, key=other_key)
                            scores.append({"events": inspected.events, "ones": inspected.ones, "trials": inspected.trials})
                        except ValueError as error:
                            scores.append({"unavailable": str(error)})
                    row["raw_counts"] = scores
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error))
                row["elapsed_seconds"] = time.monotonic() - started
                rows.append(row)
                save(private / "runs.json", rows)
                print(f"Completed {len(rows)}/32; errors: {sum('error_type' in r for r in rows)}; elapsed: {row['elapsed_seconds']:.1f}s", flush=True)
    for name, expected in plan["sdk_source_sha256"].items():
        if digest(package / name) != expected:
            raise ValueError("SDK changed during generation")
    review = [dict(review_id=r["review_id"], case=by_case[r["case"]], text=r.get("text", ""),
                   completion=r.get("completion"), error_type=r.get("error_type")) for r in rows]
    random.SystemRandom().shuffle(review)
    save(public / "blind-review.json", review)
    print("Metadata-hidden review ready; no detector or quality acceptance.", flush=True)


if __name__ == "__main__":
    main()
