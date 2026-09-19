"""Predeclared local detection feasibility screen, not deployment calibration.

Uses an installed wheel and a fresh output directory. Freezes the plan before
generation and the threshold before held-out inference. No retries or selection.
Only public/ may be exported; private keys and journals remain in the run root.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from keyprint import Keyprint


CALIBRATION = [
    "Explain how a river forms a delta and how sediment changes its shape.",
    "Describe how a public library helps a small community share knowledge.",
    "Explain how a refrigerator moves heat out of its interior.",
    "Describe the process of planning and maintaining a community garden.",
    "Explain how sound travels through air and how an echo forms.",
    "Describe how a potter turns clay into a finished ceramic bowl.",
    "Explain why different seasons occur on Earth.",
    "Describe how a small team can review a draft collaboratively.",
    "En français, explique comment le cycle de l'eau relie les océans et les nuages.",
    "En français, décris comment préparer une exposition dans une bibliothèque.",
    "En español, explica cómo las abejas contribuyen a la polinización.",
    "En español, describe cómo organizar un pequeño club de lectura.",
]
HELDOUT = [
    "Explain how roots, leaves and sunlight support the growth of a tree.",
    "Describe how a local museum might make its collection accessible to visitors.",
    "Explain how a bicycle's gears change the relationship between effort and speed.",
    "Describe how a neighborhood could organize a volunteer litter cleanup.",
    "Explain how clouds form and why some produce rain.",
    "Describe how a woodworker plans and builds a simple bookshelf.",
    "Explain how the Moon's position relates to the phases visible from Earth.",
    "Describe a careful process for proofreading and approving a newsletter.",
    "En français, explique pourquoi les feuilles de certains arbres changent de couleur en automne.",
    "En français, décris comment organiser un atelier de dessin pour débutants.",
    "En español, explica cómo el compostaje transforma los residuos vegetales.",
    "En español, describe cómo preparar una visita guiada a un museo local.",
]


def statistic(ones, trials):
    """Length-normalized bit excess; no normality or p-value interpretation."""
    if type(ones) is not int or type(trials) is not int or not 0 <= ones <= trials or trials < 1:
        raise ValueError("A statistic requires available, positive integer bit counts")
    return (2 * ones - trials) / math.sqrt(trials)


def freeze_threshold(rows):
    if not rows or any(row["split"] != "calibration" or row["condition"] != "ordinary"
                       or row.get("error") or not row.get("scores") for row in rows):
        raise ValueError("Threshold requires every ordinary calibration result")
    return max(row["scores"]["matching"]["statistic"] for row in rows)


def summarize(rows, threshold):
    result = {}
    for condition, key in (("ordinary", "matching"), ("marked", "matching"), ("marked", "other")):
        selected = [row for row in rows if row["split"] == "heldout" and row["condition"] == condition]
        available = [row["scores"][key]["statistic"] for row in selected if row.get("scores", {}).get(key)]
        result[condition + "_" + key] = {
            "attempts": len(selected), "available": len(available),
            "above_frozen_threshold": sum(value > threshold for value in available),
            "mean_statistic": sum(available) / len(available) if available else None,
            "unavailable": len(selected) - len(available),
        }
    return result


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    plan = {
        "schema": "keyprint.detection-feasibility.v1", "backend": args.backend,
        "keyprint_version": importlib.metadata.version("keyprint"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration": CALIBRATION, "heldout": HELDOUT,
        "prompt_suffix": " Respond in the requested language in about 220 words of continuous prose.",
        "max_tokens": 512, "temperature": .7, "top_k": 100,
        "statistic": "(2 * ones - trials) / sqrt(trials); not a normal p-value",
        "threshold_rule": "Maximum of all 12 ordinary calibration matching-key statistics; strict greater-than",
        "control": "One fresh independent control key; fixed for this study",
        "missing_rule": "Abort if any calibration result is unavailable; retain all held-out failures",
        "completion_rule": "Retain and score truncated outputs, with completion status; no reruns",
        "scope": "Small synthetic feasibility screen, no deployment FPR/power acceptance, no quality approval",
    }
    write_json(public / "plan.json", plan)
    keys = {name: Keyprint.new_key() for name in ("owner", "control")}
    for name, key in keys.items():
        with os.fdopen(os.open(args.output / (name + ".key"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(key)
    rows = []
    threshold = None
    status = "failed"
    failure = None
    try:
        loader = Keyprint.from_mlx if args.backend == "mlx" else Keyprint.from_transformers
        model = loader(args.model, key=keys["owner"])
        write_json(public / "identity.json", model.identity)
        for split, prompts in (("calibration", CALIBRATION), ("heldout", HELDOUT)):
            for index, prompt in enumerate(prompts):
                # Alternate held-out order in advance, never based on outputs.
                conditions = ("ordinary",) if split == "calibration" else (
                    ("ordinary", "marked") if index % 2 == 0 else ("marked", "ordinary"))
                for condition in conditions:
                    name = f"{split}-{index:02d}-{condition}"
                    row = {"id": name, "split": split, "condition": condition, "prompt": prompt}
                    started = time.monotonic()
                    try:
                        output = model.generate(prompt + plan["prompt_suffix"], condition=condition,
                                                max_tokens=plan["max_tokens"], output=args.output / name)
                        row["text"] = output.text
                        row["completion"] = output.report.get("payload", output.report).get("completion")
                        row["scores"] = {}
                        for label, key in (("matching", keys["owner"]), ("other", keys["control"])):
                            measured = model.inspect(output.text, key=key)
                            row["scores"][label] = {"ones": measured.ones, "trials": measured.trials,
                                "events": measured.events, "statistic": statistic(measured.ones, measured.trials)}
                    except Exception as exc:
                        row["error"] = type(exc).__name__
                    row["seconds"] = time.monotonic() - started
                    rows.append(row)
                    write_json(public / (name + ".json"), row)
                    print(json.dumps({k: v for k, v in row.items() if k not in ("text", "prompt")}), flush=True)
                    if split == "calibration" and row.get("error"):
                        raise RuntimeError("Calibration incomplete; no held-out generation allowed")
            if split == "calibration":
                threshold = freeze_threshold(rows)
                write_json(public / "frozen-threshold.json", {"threshold": threshold,
                    "plan_sha256": hashlib.sha256((public / "plan.json").read_bytes()).hexdigest(),
                    "calibration_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                    "comparison": "strict greater-than", "heldout_rows_seen": 0})
        status = "screen_completed" if not any(row.get("error") for row in rows) else "screen_incomplete"
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        write_json(public / "summary.json", {"status": status, "failure": failure,
            "attempts": len(rows), "threshold": threshold,
            "results": summarize(rows, threshold) if threshold is not None else None,
            "deployment_calibrated": False, "quality_acceptance": False,
            "scope": plan["scope"]})
    return 0 if status == "screen_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
