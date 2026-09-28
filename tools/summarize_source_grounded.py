"""Join frozen source-faithfulness ratings; no calibrated detector or release claim."""
import argparse
import json
from pathlib import Path

from multikey_prose_pilot import digest, save


METRICS = ("essential_facts", "supported_claims", "content", "language", "format", "full_task")


def summarize(plan, rows, ratings, rubrics, *, accept_uncertain=False):
    signature = lambda r: (r["key_slot"], r["case"], r["condition"])
    if len(rows) != 128 or [signature(r) for r in rows] != [signature(r) for r in plan["schedule"]]:
        raise ValueError("Require every planned attempt, in order")
    ids = {r["review_id"] for r in rows}
    if len(ids) != 128 or len(ratings) != 128 or {r["review_id"] for r in ratings} != ids:
        raise ValueError("Every attempt needs exactly one frozen rating")
    by_rating = {r["review_id"]: r for r in ratings}
    by_rubric = {r["id"]: r for r in rubrics}
    if len(rubrics) != 16 or set(by_rubric) != {r["case"] for r in rows}:
        raise ValueError("Require all source rubrics")
    judged = []
    for row in rows:
        rating = by_rating[row["review_id"]]
        facts = rating["fact_checks"]
        flags = rating["uncertain_fields"]
        fields = {"claims": rating["no_unsupported_claims"], "language": rating["language_pass"],
                  "format": rating["format_pass"], **{f"fact:{i}": value for i, value in enumerate(facts)}}
        if (len(facts) != len(by_rubric[row["case"]]["essential_facts"])
                or any(type(value) is not bool for value in fields.values())
                or not isinstance(flags, list) or len(flags) != len(set(flags))
                or any(flag not in fields or fields[flag] for flag in flags)
                or not isinstance(rating["reason"], str) or not rating["reason"].strip()):
            raise ValueError("Missing explicit component rating, reason or valid uncertainty flag")
        if row.get("error_type") and (any(fields.values()) or flags):
            raise ValueError("Execution failures cannot receive a passing or uncertain rating")
        if accept_uncertain:
            for field in flags: fields[field] = True
        essential = all(fields[f"fact:{i}"] for i in range(len(facts)))
        content = essential and fields["claims"]
        judged.append({"review_id": row["review_id"], "case": row["case"], "key_slot": row["key_slot"],
            "condition": row["condition"], "essential_facts": essential, "supported_claims": fields["claims"],
            "content": content, "language": fields["language"], "format": fields["format"],
            "full_task": content and fields["language"] and fields["format"]
                and row.get("completion") == "eos" and not row.get("error_type")})
    def count(items):
        return {"attempts": len(items), **{metric: sum(r[metric] for r in items) for metric in METRICS}}
    groups = {condition: count([r for r in judged if r["condition"] == condition]) for condition in ("ordinary", "marked")}
    by_case = {case: {c: count([r for r in judged if r["case"] == case and r["condition"] == c])
                     for c in groups} for case in by_rubric}
    by_key = {str(slot): {c: count([r for r in judged if r["key_slot"] == slot and r["condition"] == c])
                         for c in groups} for slot in range(4)}
    paired = {}
    index = {(r["case"], r["key_slot"], r["condition"]): r for r in judged}
    for metric in METRICS:
        counts = {name: 0 for name in ("both_pass", "ordinary_only", "marked_only", "neither_pass")}
        for case in by_rubric:
            for slot in range(4):
                ordinary, marked = (index[case, slot, c][metric] for c in groups)
                name = "both_pass" if ordinary and marked else "ordinary_only" if ordinary else "marked_only" if marked else "neither_pass"
                counts[name] += 1
        paired[metric] = counts
    return {"groups": groups, "paired": paired, "by_case": by_case, "by_key": by_key}


def diagnostics(rows):
    result = {}
    for condition in ("ordinary", "marked"):
        result[condition] = {}
        for index, key in enumerate(("matching", "next_key")):
            totals = {"available": 0, "unavailable": 0, "ones": 0, "trials": 0, "events": 0}
            for row in rows:
                if row["condition"] != condition: continue
                scores = row.get("raw_counts")
                value = scores[index] if scores is not None and len(scores) == 2 else None
                if value is None or "unavailable" in value:
                    totals["unavailable"] += 1
                    continue
                if (any(type(value[k]) is not int or value[k] < 0 for k in ("ones", "trials", "events"))
                        or value["ones"] > value["trials"]):
                    raise ValueError("Invalid raw source counts")
                totals["available"] += 1
                for field in ("ones", "trials", "events"): totals[field] += value[field]
            result[condition][key] = totals
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    public, private = args.root / "public", args.root / "private"
    read = lambda path: json.loads(path.read_text())
    commitment = read(public / "rating-commitment.json")
    for name, path in {"plan.json": public / "plan.json", "blind-review.json": private / "blind-review.json",
                       "frozen-ratings.json": private / "frozen-ratings.json"}.items():
        if digest(path) != commitment[name]: raise ValueError("Frozen review changed")
    audit = read(public / "receipt-audit.json")
    if (audit["plan_sha256"] != digest(public / "plan.json") or audit["runs_sha256"] != digest(private / "runs.json")
            or audit["blind_review_sha256"] != digest(private / "blind-review.json")):
        raise ValueError("Audited inference or review changed")
    plan, rows = read(public / "plan.json"), read(private / "runs.json")
    ratings, rubrics = read(private / "frozen-ratings.json"), read(private / "rubrics.json")
    if digest(private / "rubrics.json") != plan["rubrics_commitment"]["rubrics.json"]:
        raise ValueError("Frozen rubric changed")
    result = {"schema": "keyprint.source-grounded-fidelity.v1", "commitment": commitment,
        "strict": summarize(plan, rows, ratings, rubrics),
        "all_uncertain_fields_accepted": summarize(plan, rows, ratings, rubrics, accept_uncertain=True),
        "uncertain_fields": sum(len(r["uncertain_fields"]) for r in ratings),
        "raw_diagnostics": diagnostics(rows), "script_sha256": digest(Path(__file__)),
        "scope": "Assistant development review of sixteen predetermined English source tasks, four reused keys, one experimental Qwen3.5 profile. No independent human acceptance, calibrated detector, powered noninferiority result or universal meaning-preservation guarantee. Counts are correlated; no bitwise significance claim. Source passages and derived outputs remain private, separate from the MIT SDK. Earlier multilingual stress failures remain.",
        "quality_acceptance": False, "detector_calibrated": False, "launch_ready": False}
    save(public / "fidelity-results.json", result)
    print(json.dumps({k: result[k] for k in ("strict", "uncertain_fields", "raw_diagnostics")}, indent=2))


if __name__ == "__main__":
    main()
