"""Private source-grounded review: original passage, blank ratings, hidden conditions."""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Review template changed; inspect the source-review adaptation")
    return text.replace(old, new)


def build(source):
    rows = json.loads(source)
    if not isinstance(rows, list) or len(rows) != 128:
        raise ValueError("Require all 128 source-grounded review records")
    seen, clean = set(), []
    for row in rows:
        if set(row) != {"review_id", "case", "rubric", "text", "completion", "error_type"}:
            raise ValueError("Unexpected review metadata")
        uid, case, rubric = row["review_id"], row["case"], row["rubric"]
        if not isinstance(uid, str) or uid in seen or case["id"] != rubric["id"]:
            raise ValueError("Unique review IDs and matching rubrics required")
        seen.add(uid)
        clean.append({"review_id": uid, "text": row["text"], "completion": row["completion"],
            "error_type": row["error_type"], "title": case["instruction"],
            "case": {"id": case["id"], "language": "English", "prompt": case["instruction"],
                "source": case["source"], "facts": rubric["essential_facts"], "forbidden": rubric["qualifiers"]},
            "format_limit": rubric["format"]})
    payload = {"study_sha256": hashlib.sha256(source.encode()).hexdigest(), "rows": clean}
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('fact_review.html').read_text()
    changes = [
        ('Compare each output with the supplied facts.', 'Compare each output with its original source passage and task.'),
        ('<p class="note">Blank answers remain unreviewed.',
         '<p class="note">Private development review. Sources: Databricks Dolly 15k, CC-BY-SA-3.0 dataset and its attributed source material. Sources and derived outputs remain separate from the MIT SDK. This English-only study does not replace prior multilingual failures.</p><p class="note">Blank answers remain unreviewed.'),
        ('<p id="prompt" dir="auto"></p></details>',
         '<p id="prompt" dir="auto"></p></details><details open><summary>Original source passage</summary><div id="source" dir="auto" style="white-space:pre-wrap;overflow-wrap:anywhere"></div></details>'),
        ("const storageKey='keyprint-fact-review:'", "const storageKey='keyprint-source-review:'"),
        ("language:'',claims:'',notes:''", "language:'',claims:'',format:'',notes:''"),
        ("for(const k of ['language','claims'])", "for(const k of ['language','claims','format'])"),
        ("Boolean(r.language)&&Boolean(r.claims)", "Boolean(r.language)&&Boolean(r.claims)&&Boolean(r.format)"),
        ("$('prompt').textContent=row.case.prompt;", "$('prompt').textContent=row.case.prompt;$('source').textContent=row.case.source;"),
        ('Limit: ${row.format_limit}. Generation:', 'Requested format: ${row.format_limit} Generation:'),
        ("'unavailable'}.`;", "'unavailable'}.${row.error_type?' Execution error: '+row.error_type+'.':''}`;"),
        ("question('No unsupported claims or invented details','claims',r.claims,v=>r.claims=v,'Yes','No'))",
         "question('No unsupported claims or invented details','claims',r.claims,v=>r.claims=v,'Yes','No'),question('Requested format followed','format',r.format,v=>r.format=v,'Yes','No'))"),
        ("schema:'keyprint.human-fact-review.v1'", "schema:'keyprint.human-source-review.v1'"),
        ("no_unsupported_claims:val(r.claims),answers:", "no_unsupported_claims:val(r.claims),format_pass:val(r.format),answers:"),
        ("claims:r.claims},reason:", "claims:r.claims,format:r.format},reason:"),
        ("link.download='keyprint-fact-ratings.json'", "link.download='keyprint-source-ratings.json'"),
        ("Language and unsupported claims need separate answers.", "Language, unsupported claims and format need separate answers."),
    ]
    for old, new in changes:
        template = replace_once(template, old, new)
    return replace_once(template, '__REVIEW_DATA__', encoded)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.output.write_text(build(args.input.read_text()))
