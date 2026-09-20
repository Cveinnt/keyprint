"""Frozen minimal-pair decision screen; not a prose-quality or causal-harm claim."""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import time

SCHEMA = {"type": "object", "properties": {"allowed": {"type": "boolean"}},
          "required": ["allowed"], "additionalProperties": False}


def evaluate(text, completion, expected):
    if type(expected) is not bool:
        raise ValueError("expected answer must be boolean")
    try:
        # Duplicated keys must not silently become a valid answer.
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate field")
                value[key] = item
            return value
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        value = None
    valid = (isinstance(value, dict) and set(value) == {"allowed"}
             and type(value["allowed"]) is bool)
    return {"complete": completion == "eos", "valid": valid,
            "correct": completion == "eos" and valid and value["allowed"] is expected}


def summarize(cases, rows, repeats):
    expected = {(c["id"], r, condition) for c in cases for r in range(repeats)
                for condition in ("ordinary", "marked")}
    indexed = {}
    for row in rows:
        key = row["case"], row["repetition"], row["condition"]
        if key not in expected or key in indexed:
            raise ValueError("duplicate or undeclared attempt")
        indexed[key] = row
    stats = {condition: {"attempts": 0, "correct": 0, "errors": 0} for condition in ("ordinary", "marked")}
    paired = Counter()
    for row in rows:
        stat = stats[row["condition"]]
        stat["attempts"] += 1
        stat["correct"] += bool(row.get("checks", {}).get("correct"))
        stat["errors"] += "error_type" in row
    for case in cases:
        for r in range(repeats):
            a, b = indexed.get((case["id"], r, "ordinary")), indexed.get((case["id"], r, "marked"))
            if a is None or b is None:
                continue
            good_a, good_b = a.get("checks", {}).get("correct", False), b.get("checks", {}).get("correct", False)
            paired["both_correct" if good_a and good_b else "ordinary_only_correct" if good_a else
                   "marked_only_correct" if good_b else "neither_correct"] += 1
    complete = set(indexed) == expected
    return {"complete": complete, "expected_attempts": len(expected), "conditions": stats,
            "paired_outcomes": dict(paired),
            "fixed_screen_pass": complete and all(row.get("checks", {}).get("correct") for row in rows),
            "quality_acceptance": False,
            "scope": "Fixed synthetic decisions under a boolean JSON grammar; repeated sources, one key, independent samples. Not prose fidelity, deployment coverage or causal watermark harm."}


def save(output, plan, rows):
    report = {"plan_sha256": hashlib.sha256((output / "public/plan.json").read_bytes()).hexdigest(),
              "summary": summarize(plan["cases"], rows, plan["repeats"]), "runs": rows}
    (output / "public/results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    sections = []
    for case in plan["cases"]:
        records = [row for row in rows if row["case"] == case["id"]]
        sections.append('<section><h2>' + html.escape(case["id"]) + '</h2><p>' + html.escape(case["prompt"]) +
            '</p><p>Expected: ' + str(case["expected"]).lower() + '</p><table><thead><tr><th>Repetition</th><th>Condition</th><th>Exact output</th><th>Result</th></tr></thead><tbody>' +
            ''.join('<tr><td>' + str(r["repetition"] + 1) + '</td><td>' + r["condition"] + '</td><td><pre>' +
                    html.escape(r.get("text", r.get("error_type", ""))) + '</pre></td><td>' +
                    ("Correct" if r.get("checks", {}).get("correct") else "Failed") + '</td></tr>' for r in records) + '</tbody></table></section>')
    (output / "public/comparison.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Keyprint decision comparison</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;background:#f7f5ee;color:#292923;font:18px/1.5 Georgia,serif}section{border-top:1px solid #ccc;padding:20px 0}table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:36px}</style><h1>Approval, negation and deadlines</h1><p>Every declared ordinary and marked attempt, without retries. The grammar permits either answer. This narrow decision screen does not approve rewriting, detection, or production quality.</p><pre>''' + html.escape(json.dumps(report["summary"], indent=2)) + '</pre>' + ''.join(sections) + '</html>')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    cases_file = Path(__file__).with_name('decision_quality_cases.json')
    cases = json.loads(cases_file.read_text())
    assert len(cases) == 16 and len({c['id'] for c in cases}) == 16
    assert all(type(c['expected']) is bool for c in cases)
    args.output.mkdir(mode=0o700)
    (args.output / 'public').mkdir()
    from keyprint import Keyprint
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / 'owner.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(key)
    with Keyprint.from_mlx(args.model, key=key) as candidate:
        plan = {'cases': cases, 'repeats': args.repeats, 'schema': SCHEMA, 'max_tokens': 32,
                'identity': candidate.identity, 'key_commitment': hashlib.sha256(key).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'cases_sha256': hashlib.sha256(cases_file.read_bytes()).hexdigest(),
                'execution': 'reference', 'ordering': 'Alternate first condition by case index plus repetition',
                'selection': 'All 16 declared cases, four languages, eight positive/negative minimal pairs',
                'primary': 'Every declared output completes and returns exactly the correct boolean; any failure fails this narrow screen',
                'failure_rule': 'No retries, replacements, edits, answer-forcing schema or excluded attempts',
                'scope': __doc__}
        (args.output / 'public/plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
        rows = []
        save(args.output, plan, rows)
        for repetition in range(args.repeats):
            for i, case in enumerate(cases):
                order = ('ordinary', 'marked') if (i + repetition) % 2 == 0 else ('marked', 'ordinary')
                for condition in order:
                    row = {'case': case['id'], 'repetition': repetition, 'condition': condition}
                    started = time.monotonic()
                    try:
                        result = candidate.generate(case['prompt'], condition=condition, max_tokens=32,
                            json_schema=SCHEMA, output=args.output / f"{case['id']}-{repetition}-{condition}")
                        completion = result.report['payload']['completion']
                        row.update(text=result.text, completion=completion, usage=result.report['usage'],
                                   checks=evaluate(result.text, completion, case['expected']))
                    except Exception as exc:
                        row['error_type'] = type(exc).__name__
                    row['seconds'] = time.monotonic() - started
                    rows.append(row)
                    report = save(args.output, plan, rows)
                    print(len(rows), case['id'], condition, row.get('checks', row.get('error_type')), flush=True)
        print(json.dumps(report['summary']), flush=True)
        return int(not report['summary']['fixed_screen_pass'])


if __name__ == '__main__':
    raise SystemExit(main())
