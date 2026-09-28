"""Summarize complete native replay telemetry without selecting successful paths."""
import argparse
import json
import math
from pathlib import Path
import statistics

from diagnose_source_sampling import aggregate
from multikey_prose_pilot import digest, save


def summarize(rows, replay, traces):
    """Require all original attempts; a partial/failed replay cannot become a cohort."""
    if len(rows) != 128 or len(replay) != 128:
        raise ValueError("Require all 128 original attempts and replay outcomes")
    ids = [r["review_id"] for r in rows]
    signatures = {(r["case"], r["key_slot"], r["condition"]) for r in rows}
    cases = {r["case"] for r in rows}
    expected = {(c, k, arm) for c in cases for k in range(4) for arm in ("ordinary", "marked")}
    if (len(set(ids)) != 128 or set(traces) != set(ids) or len(cases) != 16
            or signatures != expected):
        raise ValueError("Missing, duplicate or unplanned case/key/condition")
    fields = ("review_id", "case", "key_slot", "condition")
    for row, result in zip(rows, replay):
        steps = traces[row["review_id"]]
        if (any(row[k] != result[k] for k in fields) or result["passed"] is not True
                or result["matched_steps"] != row["tokens"]
                or result["model_calls"] != row["model_calls"]
                or len(steps) != row["tokens"] or not steps
                or any(type(s["selected_is_eos"]) is not bool for s in steps)
                or not steps[-1]["selected_is_eos"]
                or any(s["selected_is_eos"] for s in steps[:-1])
                or any(not math.isfinite(v) for s in steps for v in s.values())
                or aggregate(steps) != result["diagnostics"]):
            raise ValueError("Incomplete, failed, reordered or changed replay telemetry")
    groups = {}
    for arm in ("ordinary", "marked"):
        trajectories = [traces[r["review_id"]] for r in rows if r["condition"] == arm]
        steps = [s for trajectory in trajectories for s in trajectory]
        combined = aggregate(steps)
        eos = combined.pop("selected_eos")
        numeric = list(combined["mean"])
        groups[arm] = {"attempts": len(trajectories), "token_weighted": combined,
            "equal_output_mean": {k: statistics.mean(statistics.mean(s[k] for s in t)
                for t in trajectories) for k in numeric},
            "ending_tokens": {"count": len(eos), "mean": {k: statistics.mean(s[k] for s in eos)
                for k in numeric}},
            "eos_mass_max_absolute_change": max(abs(s["eos_effective_mass"] - s["eos_base_mass"]) for s in steps),
            "selected_base_probability_median": statistics.median(s["selected_base_probability"] for s in steps),
            "selected_base_rank_median": statistics.median(s["selected_base_rank"] for s in steps)}
    return {"schema": "keyprint.source-sampling-summary.v1", "groups": groups,
        "scope": "All original paths. Within-step distribution changes share the same recorded prefix. Between-arm paths differ; neither comparison isolates semantic causation. Tokens, tasks and reused keys are dependent. No detector or quality acceptance.",
        "quality_acceptance": False, "launch_ready": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    public, private = root / "public", root / "private"
    read = lambda f: json.loads(f.read_text())
    result_file = public / "sampling-diagnosis-results.json"
    result = read(result_file)  # No progress-file fallback.
    plan = read(public / "sampling-diagnosis-plan.json")
    rows = read(private / "runs.json")
    if (result["plan"] != plan or result["attempts"] != 128 or result["verified"] != 128
            or plan["runs_sha256"] != digest(private / "runs.json")
            or plan["study_plan_sha256"] != digest(public / "plan.json")
            or plan["script_sha256"] != digest(Path(__file__).with_name("diagnose_source_sampling.py"))
            or result["matched_steps"] != sum(r["tokens"] for r in rows)):
        raise ValueError("Require complete final replay bound to original study")
    folder = private / "sampling-diagnosis"
    traces = {r["review_id"]: read(folder / (r["review_id"] + ".json")) for r in rows}
    summary = summarize(rows, result["rows"], traces)
    summary["commitment"] = {"results_sha256": digest(result_file), "script_sha256": digest(Path(__file__)),
        "diagnostic_helper_sha256": digest(Path(__file__).with_name("diagnose_source_sampling.py")),
        "trace_sha256": {i: digest(folder / (i + ".json")) for i in traces}}
    save(public / "sampling-diagnosis-summary.json", summary)


if __name__ == "__main__":
    main()
