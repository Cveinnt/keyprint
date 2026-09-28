"""Reconcile ordinary upstream model attempts and frozen assistant judgments."""
import argparse
import json
from pathlib import Path
import re

from validate_model_baseline import digest, reconcile, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path, required=True)
    args = p.parse_args()
    public, private = args.root / "public", args.root / "private"
    plan = json.loads((public / "plan.json").read_text())
    commitment = json.loads((public / "rating-commitment.json").read_text())
    for name in ("plan.json", "blind-review.json", "frozen-ratings.json"):
        if digest(public / name) != commitment[name]:
            raise ValueError("Review commitment changed")
    for name, expected in [("validate_model_baseline.py", plan["script_sha256"]),
                           ("long_fidelity_cases.json", plan["cases_sha256"])]:
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Registered source changed")
    if args.model.name != plan["revision"]:
        raise ValueError("Wrong snapshot")
    for name, expected in plan["assets_sha256"].items():
        if digest(args.model / name) != expected:
            raise ValueError("Model assets changed")
    # Reconstruct the exact upstream loader policy used by mlx_lm.load.
    # A nested text_config EOS can differ from the tokenizer definition.
    from mlx_lm.utils import load_tokenizer
    config = json.loads((args.model / "config.json").read_text())
    tokenizer = load_tokenizer(args.model, {"trust_remote_code": False, "local_files_only": True},
                               eos_token_ids=config.get("eos_token_id", None))
    eos = sorted(tokenizer.eos_token_ids)
    rows = json.loads((private / "runs.json").read_text())
    ratings = json.loads((public / "frozen-ratings.json").read_text())
    blind = json.loads((public / "blind-review.json").read_text())
    signature = lambda r: (r["case"], r["repetition"], r["seed"])
    if (len(rows) != len(plan["schedule"]) or len({signature(r) for r in rows}) != len(rows)
            or {signature(r) for r in rows} != {signature(r) for r in plan["schedule"]}):
        raise ValueError("Missing, duplicate or undeclared attempts")
    ids = {r["review_id"] for r in rows}
    if (len(ids) != len(rows) or any(len(x) != len(rows) for x in (ratings, blind))
            or any({r["review_id"] for r in x} != ids for x in (ratings, blind))):
        raise ValueError("Review IDs do not reconcile")
    by_case = {c["id"]: c for c in plan["cases"]}
    labels = {r["review_id"]: r for r in ratings}
    views = {r["review_id"]: r for r in blind}
    judged = []
    for row in rows:
        uid = row["review_id"]
        folder = private / uid
        if digest(folder / "events.json") != row["events_sha256"]:
            raise ValueError("Generation events changed")
        events = json.loads((folder / "events.json").read_text())
        if not row.get("error_type"):
            decoded = lambda x: tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            actual = reconcile(events, decoded, eos, plan["settings"]["max_tokens"])
            if any(row[k] != v for k, v in actual.items()) or row["model_forward_calls"] <= 0:
                raise ValueError("Native events, text, tokens or forward evidence differ")
            prompt = json.loads((folder / "prompt.json").read_text())
            expected_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": by_case[row["case"]]["prompt"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if (prompt["ids"] != expected_ids or prompt["seed"] != row["seed"]
                    or len(prompt["ids"]) != row["prompt_tokens"]
                    or prompt["random_state_before"] == row["random_state_after"]):
                raise ValueError("Prompt, seed or random-state evidence differs")
        case, label, view = by_case[row["case"]], labels[uid], views[uid]
        if view["text"] != row["text"] or view["case"] != case:
            raise ValueError("Reviewed text or source facts differ")
        facts = label["fact_checks"]
        if (len(facts) != len(case["facts"]) or any(type(v) is not bool for v in facts)
                or any(type(label[k]) is not bool for k in ("language_pass", "prose_pass", "no_unsupported_claims", "borderline"))
                or not label["reason"]):
            raise ValueError("Incomplete ratings")
        unit = "words" if "max_words" in case else "chars"
        text = row.get("text", "")
        length = len(text.split()) if unit == "words" else len(text)
        paragraphs = len(re.split(r"\n\s*\n", text.strip())) if text.strip() else 0
        execution = not row.get("error_type") and row.get("completion") == "eos"
        content = bool(text) and not row.get("error_type") and all(facts) and label["no_unsupported_claims"]
        format_ok = case["min_"+unit] <= length <= case["max_"+unit] and paragraphs == case["paragraphs"] and label["prose_pass"]
        judged.append({"review_id": uid, "length": length, "paragraphs": paragraphs,
                       "execution_pass": execution, "content_pass": content,
                       "language_pass": label["language_pass"], "format_pass": format_ok,
                       "task_pass": execution and content and label["language_pass"] and format_ok,
                       "conservative_content_pass": content or label["borderline"],
                       "conservative_task_pass": execution and (content or label["borderline"]) and label["language_pass"] and format_ok})
    names = ("execution_pass", "content_pass", "language_pass", "format_pass", "task_pass",
             "conservative_content_pass", "conservative_task_pass")
    result = {"schema": "keyprint.model-baseline.audit.v1", "attempts": len(rows),
              "counts": {k: sum(r[k] for r in judged) for k in names},
              "tokens": sum(r.get("tokens", 0) for r in rows),
              "model_forward_calls": sum(r["model_forward_calls"] for r in rows),
              "runs": rows, "judgments": judged, "audit_script_sha256": digest(Path(__file__)),
              "scope": plan["scope"], "quality_acceptance": False, "sdk_compatibility": False,
              "termination_policy": {"resolved_upstream_eos_ids": eos,
                                     "nested_config_eos": config.get("text_config", {}).get("eos_token_id"),
                                     "top_level_config_eos": config.get("eos_token_id"),
                                     "source": "mlx_lm.utils.load tokenizer resolution; no generation rerun or text change"},
              "interpretation": "Assistant development screen. Original ratings and all conservative flags retained. No watermarked samples or calibrated detector; cannot estimate watermark-caused harm."}
    save(public / "results.json", result)
    print(json.dumps({k: result[k] for k in ("attempts", "counts", "tokens", "model_forward_calls")}, indent=2))


if __name__ == "__main__":
    main()
