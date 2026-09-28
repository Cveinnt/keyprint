"""Long source-grounded prose tasks through the unchanged public SDK.

Separately rate content, requested language and format. No semantic guarantees,
human acceptance or scientific clue closure follows from this development set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import uuid

from multikey_prose_pilot import REVISION, digest, save


def schedule(cases):
    return [{'key_slot': slot, 'case': case['id'], 'condition': condition}
            for slot in range(4) for index, case in enumerate(cases)
            for condition in (('ordinary', 'marked') if (slot + index) % 2 == 0 else ('marked', 'ordinary'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model.name != REVISION:
        parser.error('Use pinned model snapshot')
    from keyprint import Keyprint
    from keyprint.integrity import verify
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from layer_likelihood import extract
    from weighted_null import reference_tail
    cases_path = Path(__file__).with_name('long_fidelity_cases.json')
    cases = json.loads(cases_path.read_text())
    if len(cases) != 4 or len({c['id'] for c in cases}) != 4:
        raise ValueError('Require all four unique tasks')
    keys = [(args.prior / f'private/key-{i}').read_bytes() for i in range(4)]
    args.output.mkdir(mode=0o700)
    public, private = args.output / 'public', args.output / 'private'
    public.mkdir(); private.mkdir(mode=0o700)
    plan = {'schema': 'keyprint.long-fidelity.v1', 'cases': cases,
            'schedule': schedule(cases), 'max_tokens': 1024, 'temperature': .7, 'top_k': 100,
            'model_revision': REVISION, 'script_sha256': digest(Path(__file__)),
            'cases_sha256': digest(cases_path), 'engine_integrity': verify(),
            'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                              ('multikey_prose_pilot.py', 'layer_likelihood.py', 'weighted_null.py')},
            'prior_plan_sha256': digest(args.prior / 'public/plan.json'),
            'key_sha256': [hashlib.sha256(key).hexdigest() for key in keys],
            'key_policy': 'All four existing keys, including known failure; no selection or rotation',
            'failure_policy': '32 attempts, no retries/replacement/rewriting; retain errors and caps',
            'review': 'Freeze per-fact and forbidden-addition content ratings, language ratings and mechanical format results before revealing conditions/keys/scores. Assistant review; not independent human acceptance.',
            'primary': 'Require registered minimum/maximum length and exactly three prose paragraphs separately from content. Preserve all failures. Joint outcome is full-task pass AND matching hit AND no other-key hit. Separate content, language, format and execution failures; per-task/paired/key counts. Descriptive longer-task development on deliberately selected prior-failure domains, not a powered noninferiority or causal study.',
            'detector': 'Unchanged weighted 30-layer tail <= .005 per key for matching/next-slot keys; nominal two-key 1%, uncalibrated. Same nominal threshold as short study, no length-based cutoff change. Detection and full-task pass must hold on the same output; report their intersection.',
            'scope': 'Four new longer prompts built from previously examined source-fact domains, four languages, one Qwen/MLX model and four reused keys. No open-domain factual, framework, scientific or launch acceptance.'}
    save(public / 'plan.json', plan)
    binding = runtime_binding(max_steps=2048)
    by_case = {c['id']: c for c in cases}
    rows = []
    for slot, key in enumerate(keys):
        with Keyprint.from_mlx(args.model, key=key, temperature=.7, top_k=100) as wm:
            for attempt in [a for a in plan['schedule'] if a['key_slot'] == slot]:
                row = dict(attempt, review_id=uuid.uuid4().hex[:12])
                try:
                    result = wm.generate(by_case[row['case']]['prompt'], condition=row['condition'],
                                         max_tokens=1024, trace=True, output=private / row['review_id'])
                    payload = result.report['payload']
                    if ''.join(t.text for t in result.trace) != result.text:
                        raise ValueError('Token trace differs from returned text')
                    row.update(text=result.text, completion=payload['completion'],
                               tokens=len(payload['committed_token_ids']), exact_rendering=True)
                    scores = [reference_tail(extract(binding, result.text, keys[k]))
                              for k in (slot, (slot + 1) % 4)]
                    row.update(scores=scores, matching_hit=scores[0]['reference_tail'] <= .005,
                               other_hit=scores[1]['reference_tail'] <= .005)
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error))
                rows.append(row)
                save(private / 'runs.json', rows)
                print(f'Completed {len(rows)}/32; errors: {sum("error_type" in r for r in rows)}', flush=True)
    review = []
    for row in rows:
        case = by_case[row['case']]
        text = row.get('text', '')
        review.append(dict(review_id=row['review_id'], case=case, text=text,
                           completion=row.get('completion'), error_type=row.get('error_type'),
                           words=len(text.split()), chars=len(text)))
    random.SystemRandom().shuffle(review)
    save(public / 'blind-review.json', review)
    print('Metadata-hidden factual review ready.', flush=True)


if __name__ == '__main__':
    main()
