"""All-key source-grounded paired inference; inputs/ratings remain private."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

from multikey_prose_pilot import digest, save


def schedule(cases):
    if len(cases) != 16 or len({c["id"] for c in cases}) != 16:
        raise ValueError("All sixteen unique frozen cases are required")
    return [{"case": c["id"], "key_slot": slot, "condition": condition}
            for slot in range(4) for index, c in enumerate(cases)
            for condition in (("ordinary", "marked") if (slot + index) % 2 == 0 else ("marked", "ordinary"))]


def load_inputs(root):
    selection = json.loads((root / "public/selection.json").read_text())
    commitment = json.loads((root / "public/rubrics-commitment.json").read_text())
    for name in ("cases.json", "rubrics.json"):
        if digest(root / "private" / name) != commitment[name]:
            raise ValueError("Frozen input or rubric changed")
    if digest(root / "private/cases.json") != selection["private_cases_sha256"]:
        raise ValueError("Selected source cases changed")
    cases = json.loads((root / "private/cases.json").read_text())
    rubrics = json.loads((root / "private/rubrics.json").read_text())
    if ({r["id"] for r in rubrics} != {c["id"] for c in cases} or len(rubrics) != len(cases)
            or any(not r["essential_facts"] or not r["format"] or not r["qualifiers"] for r in rubrics)):
        raise ValueError("Missing, duplicate or incomplete source rubric")
    schedule(cases)
    return cases, rubrics, selection, commitment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases, rubrics, selection, commitment = load_inputs(args.selection)
    keys = [(args.prior / f"private/key-{i}").read_bytes() for i in range(4)]
    if any(len(k) != 32 for k in keys):
        raise ValueError("Four original keys required")
    import keyprint
    from keyprint import Keyprint
    from keyprint.integrity import verify
    package = Path(keyprint.__file__).parent
    args.output.mkdir(mode=0o700)
    public, private = args.output / "public", args.output / "private"
    public.mkdir(); private.mkdir(mode=0o700)
    # Private copying binds source/criteria to this exact attempt set.
    save(private / "cases.json", cases)
    save(private / "rubrics.json", rubrics)
    plan = {"schema": "keyprint.source-grounded-run.v1", "selection": selection,
            "rubrics_commitment": commitment, "schedule": schedule(cases),
            "settings": {"execution": "experimental-wide", "temperature": .7, "top_k": 100, "max_tokens": 768},
            "script_sha256": digest(Path(__file__)),
            "helper_sha256": {"multikey_prose_pilot.py": digest(Path(__file__).with_name("multikey_prose_pilot.py"))},
            "sdk_source_sha256": {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))},
            "key_sha256": [hashlib.sha256(k).hexdigest() for k in keys], "engine_integrity": verify(),
            "failure_policy": "128 planned attempts, no retries/replacement/rewriting. Retain every error and cap. Never select a passing ordinary answer before its marked counterpart.",
            "review": commitment["criteria"],
            "scope": "English source-grounded development evaluation on sixteen predetermined public-dataset instructions, four existing keys, one pinned experimental Qwen3.5 profile. Not known training-held-out. No detector threshold, causal/noninferiority acceptance or release readiness. Existing stress failures retained.",
            "distribution": "Private dataset sources, prompts and source-derived results; separate from MIT SDK. Public manifest identifies provenance/hashes without source text."}
    save(public / "plan.json", plan)
    by_case, by_rubric = ({r["id"]: r for r in collection} for collection in (cases, rubrics))
    rows = []
    for slot, key in enumerate(keys):
        with Keyprint.from_mlx(args.model, key=key, execution="experimental-wide", temperature=.7, top_k=100) as wm:
            ip = public / "identity.json"
            if ip.exists() and json.loads(ip.read_text()) != wm.identity:
                raise ValueError("Profile identity changed between keys")
            if not ip.exists(): save(ip, wm.identity)
            for attempt in [r for r in plan["schedule"] if r["key_slot"] == slot]:
                row = dict(attempt, review_id=uuid.uuid4().hex[:12])
                started = time.monotonic()
                try:
                    generated = wm.generate(by_case[row["case"]]["prompt"], condition=row["condition"],
                        max_tokens=768, trace=True, output=private / row["review_id"])
                    report = generated.report
                    row.update(text=generated.text, completion=report["completion"],
                        tokens=len(report["committed_token_ids"]), model_calls=report["model_calls"],
                        report_sha256=digest(generated.artifacts / "report.json"),
                        journal_sha256=digest(generated.artifacts / "journal.jsonl"))
                    if "".join(token.text for token in generated.trace) != generated.text:
                        raise ValueError("Trace differs from emitted text")
                    scores = []
                    for other in (key, keys[(slot + 1) % 4]):
                        try:
                            result = wm.inspect(generated.text, key=other)
                            scores.append({"events": result.events, "ones": result.ones, "trials": result.trials})
                        except ValueError as error:
                            scores.append({"unavailable": str(error)})
                    row["raw_counts"] = scores
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error))
                row["elapsed_seconds"] = time.monotonic() - started
                rows.append(row)
                save(private / "runs.json", rows)
                print(f"Completed {len(rows)}/128; errors {sum('error_type' in r for r in rows)}; elapsed {row['elapsed_seconds']:.1f}s", flush=True)
    for name, expected in plan["sdk_source_sha256"].items():
        if digest(package / name) != expected:
            raise ValueError("SDK changed during inference")
    review = [{"review_id": r["review_id"], "case": by_case[r["case"]], "rubric": by_rubric[r["case"]],
               "text": r.get("text", ""), "completion": r.get("completion"), "error_type": r.get("error_type")}
              for r in rows]
    random.SystemRandom().shuffle(review)
    save(private / "blind-review.json", review)
    print("128-attempt source-grounded review ready; conditions/keys/counts hidden; no acceptance claim.", flush=True)


if __name__ == "__main__":
    main()
