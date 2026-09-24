"""Fixed four-key follow-up to the September 23 prose pilot."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

REVISION = "545dc4251c05440727734bcd94334791f6ab0192"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def repetition_summary(plan, rows):
    expected = {(slot, case['id'], rep, condition)
                for slot in range(plan['keys']) for case in plan['cases']
                for rep in range(plan['repeats'])
                for condition in ('ordinary', 'marked')}
    indexed = {}
    ids = set()
    for row in rows:
        attempt = (row['key_slot'], row['case'], row['repetition'], row['condition'])
        if attempt not in expected or attempt in indexed or row['review_id'] in ids:
            raise ValueError('Duplicate or undeclared attempt')
        indexed[attempt] = row
        ids.add(row['review_id'])
    groups = []
    for slot in range(plan['keys']):
        for case in plan['cases']:
            for condition in ('ordinary', 'marked'):
                group = [r for key, r in indexed.items()
                         if key[0] == slot and key[1] == case['id'] and key[3] == condition]
                texts = [r['text'] for r in group if not r.get('error_type') and r.get('completion') == 'eos']
                counts = Counter(texts)
                groups.append({'key_slot': slot, 'case': case['id'], 'condition': condition,
                               'attempts': len(group), 'completed': len(texts),
                               'distinct_texts': len(counts),
                               'largest_identical_group': max(counts.values(), default=0)})
    return {'complete': set(indexed) == expected, 'attempts': len(rows), 'groups': groups,
            'scope': 'Exact-text repetition counts on three reused prompts and four keys; not quality acceptance or a general diversity estimate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model.name != REVISION:
        parser.error('Use the registered pinned model snapshot directory')
    from keyprint import Keyprint
    cases_path = Path(__file__).with_name('prose_quality_cases.json')
    cases = json.loads(cases_path.read_text())
    out = args.output
    out.mkdir(mode=0o700)
    (out / 'private').mkdir(mode=0o700)
    (out / 'public').mkdir()
    plan = {'keys': 4, 'repeats': 3, 'cases': cases, 'max_tokens': 192,
            'temperature': 0.7, 'top_k': 100, 'model_revision': REVISION,
            'backend': 'MLX reference', 'script_sha256': digest(Path(__file__)),
            'cases_sha256': digest(cases_path),
            'ordering': 'Key slot, repetition, case; first condition alternates with their index sum',
            'key_policy': 'Four independent fresh private keys; retain every key and output, no key selection',
            'failure_policy': 'Retain all 72 attempts including errors; no retries or replacements',
            'review': 'Assistant ratings against unchanged rubrics frozen before joining condition/key mapping; not independent human review',
            'primary': 'Per-key exact-output repetition and descriptive task counts, no general quality acceptance'}
    save(out / 'public/plan.json', plan)
    rows = []
    for slot in range(plan['keys']):
        key = Keyprint.new_key()
        with os.fdopen(os.open(out / f'private/key-{slot}', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
            stream.write(key)
        with Keyprint.from_mlx(args.model, key=key, temperature=.7, top_k=100) as wm:
            for rep in range(plan['repeats']):
                for i, case in enumerate(cases):
                    order = ('ordinary', 'marked') if (slot + rep + i) % 2 == 0 else ('marked', 'ordinary')
                    for condition in order:
                        row = {'key_slot': slot, 'case': case['id'], 'repetition': rep,
                               'condition': condition, 'review_id': uuid.uuid4().hex[:12]}
                        start = time.perf_counter()
                        try:
                            result = wm.generate(case['prompt'], condition=condition, max_tokens=192,
                                                 trace=True, output=out / 'private' / row['review_id'])
                            payload = result.report.get('payload', result.report)
                            if ''.join(t.text for t in result.trace) != result.text:
                                raise ValueError('Trace does not reconstruct generated text')
                            row.update(text=result.text, completion=payload['completion'],
                                       tokens=len(result.trace), exact_rendering=True)
                        except Exception as error:
                            row['error_type'] = type(error).__name__
                        row['seconds'] = time.perf_counter() - start
                        rows.append(row)
                        save(out / 'private/runs.json', rows)
                        print(f'Completed {len(rows)}/72; errors: {sum("error_type" in r for r in rows)}', flush=True)
    by_case = {c['id']: c for c in cases}
    reviews = [{'review_id': r['review_id'], 'case': r['case'],
                'prompt': by_case[r['case']]['prompt'], 'rubric': by_case[r['case']]['rubric'],
                'text': r.get('text', ''), 'completion': r.get('completion'),
                'word_count': len(r.get('text', '').split()), 'error_type': r.get('error_type')}
               for r in rows]
    random.SystemRandom().shuffle(reviews)
    save(out / 'public/blind-review.json', reviews)
    # Mapping stays private until the ratings are frozen.
    save(out / 'private/repetition-summary.json', repetition_summary(plan, rows))
    print('Condition/key-hidden review ready', flush=True)


if __name__ == '__main__':
    main()
