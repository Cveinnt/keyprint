"""Join frozen assistant ratings without importing old-profile detection rules."""
import argparse
import copy
import json
from pathlib import Path

from multikey_prose_pilot import digest, save
from summarize_long_fidelity import summarize


def summary(plan, rows, ratings):
    # Reuse exactly the existing content/language/format definitions, then remove
    # unavailable detection fields instead of reporting missing evidence as zero.
    result = summarize(plan, rows, ratings)
    for group in result["groups"].values():
        for name in ("matching_hits", "other_hits", "joint_pass"):
            group.pop(name)
    result["paired"].pop("joint_pass")
    for item in [*result["by_case"], *result["judgments"].values()]:
        item.pop("joint_pass")
    result["interpretation"] = "Assistant development review on four long tasks and reused keys. No independent human acceptance, causal/noninferiority estimate or new-profile detector threshold. Matching/control counts are descriptive only."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    public, private = args.root / "public", args.root / "private"
    commitment = json.loads((public / "rating-commitment.json").read_text())
    for name in ("plan.json", "blind-review.json", "frozen-ratings.json"):
        if digest(public / name) != commitment[name]:
            raise ValueError("Frozen review changed")
    plan = json.loads((public / "plan.json").read_text())
    rows = json.loads((private / "runs.json").read_text())
    ratings = json.loads((public / "frozen-ratings.json").read_text())
    views = json.loads((public / "blind-review.json").read_text())
    by_id = {v["review_id"]: v for v in views}
    by_case = {c["id"]: c for c in plan["cases"]}
    if len(by_id) != len(rows) or {r["review_id"] for r in rows} != set(by_id):
        raise ValueError("Missing/duplicate/unknown blind review rows")
    for row in rows:
        view = by_id[row["review_id"]]
        if view["text"] != row.get("text", "") or view["case"] != by_case[row["case"]]:
            raise ValueError("Reviewed text/source differs")
    strict = summary(plan, rows, ratings)
    generous = copy.deepcopy(ratings)
    for label in generous:
        if type(label["borderline"]) is not bool:
            raise ValueError("Every conservative flag must be explicit")
        if label["borderline"]:
            label["fact_checks"] = [True] * len(label["fact_checks"])
            label["no_unsupported_claims"] = True
    result = {"schema": "keyprint.wide-fidelity.v1", "strict": strict,
              "all_conservative_flags_accepted": summary(plan, rows, generous),
              "conservative_flags": sum(r["borderline"] for r in ratings),
              "runs": rows, "ratings": ratings, "commitment": commitment,
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                 ("summarize_wide_fidelity.py", "summarize_long_fidelity.py")},
              "detector_calibrated": False, "quality_acceptance": False, "launch_ready": False}
    save(public / "fidelity-results.json", result)
    print(json.dumps({"strict": strict["groups"], "paired": strict["paired"],
                     "conservative_flags": result["conservative_flags"],
                     "all_flags_accepted": result["all_conservative_flags_accepted"]["groups"]}, indent=2))


if __name__ == "__main__":
    main()
