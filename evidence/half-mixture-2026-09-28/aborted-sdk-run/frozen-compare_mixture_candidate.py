"""Frozen development screen of half-mixture fidelity, repetition and signal.

Not release qualification. Existing keys are deliberately retained, including a
known failure. No retries, favorable-key selection or detector threshold search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import uuid

from mixture_candidate import MixturePublicCandidate

REVISION = '545dc4251c05440727734bcd94334791f6ab0192'
ARMS = ('ordinary', 'reference', 'half_mixture')
LONG_CASES = [
    {'id': 'weather_english', 'prompt': 'Write 150 to 200 words in English, in three short paragraphs, explaining the difference between weather and climate for a general reader. Explain that weather describes short-term atmospheric conditions, climate describes long-term patterns, and a single cold day does not establish a long-term climate trend. Do not invent a named place, person, measured statistic or a specific date.',
     'rubric': 'English, 150 to 200 whitespace-separated words in three prose paragraphs. Correct short-term weather versus long-term climate distinction; a cold day alone does not establish a climate trend. No invented named places, people, statistics or dates; no materially false scientific explanation.'},
    {'id': 'library_spanish', 'prompt': 'Escribe entre 150 y 200 palabras en español, en tres párrafos breves, explicando cómo una biblioteca puede organizar préstamos de libros. Presenta las recomendaciones como posibilidades, no como hechos de una biblioteca real. Explica que se registra qué libro se presta, quién lo recibe y cuándo debe devolverlo; que una renovación depende de las reglas de la biblioteca; y que conviene proteger los datos personales. No inventes nombres, fechas específicas, precios ni multas.',
     'rubric': 'Spanish, 150 to 200 whitespace-separated words in three prose paragraphs. Presents recommendations rather than facts about a real library. Covers book, borrower, due date, rule-dependent renewal and protection of personal data. No invented names, specific dates, prices or fines; no materially incorrect claims.'},
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def schedule():
    attempts = []
    for slot in range(4):
        tasks = [('french', 0), ('approval', 0), ('french', 1), (LONG_CASES[slot % 2]['id'], 0)]
        for index, (case, repetition) in enumerate(tasks):
            offset = (slot + index) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                attempts.append(dict(key_slot=slot, case=case, repetition=repetition, arm=arm))
    return attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model.name != REVISION:
        parser.error('Use the pinned snapshot')
    from keyprint import Keyprint
    from keyprint._engine.research.keyprint_v3_public_api_rc2 import PublicCandidate
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from layer_likelihood import extract
    from weighted_null import reference_tail
    case_file = Path(__file__).with_name('prose_quality_cases.json')
    cases = [c for c in json.loads(case_file.read_text()) if c['id'] in ('french', 'approval')] + LONG_CASES
    by_case = {c['id']: c for c in cases}
    args.output.mkdir(mode=0o700)
    private, public = args.output / 'private', args.output / 'public'
    private.mkdir(mode=0o700); public.mkdir()
    plan = {'schema': 'keyprint.half-mixture-development.v1', 'model_revision': REVISION,
            'script_sha256': digest(Path(__file__)), 'source_sha256': {
                name: digest(Path(__file__).with_name(name)) for name in
                ('mixture_candidate.py', 'prose_quality_cases.json', 'weighted_null.py', 'layer_likelihood.py')},
            'prior_plan_sha256': digest(args.prior / 'public/plan.json'),
            'keys': [0, 1, 2, 3], 'cases': cases, 'schedule': schedule(),
            'temperature': .7, 'top_k': 100, 'short_max_tokens': 192, 'long_max_tokens': 768,
            'key_policy': 'Every key from the concentration audit; no rotation, selection or regeneration',
            'failure_policy': 'Retain every attempt, cap and error; no retries or output replacement',
            'detector': {'rule': 'Unchanged weighted_null.reference_tail, matching plus next-slot key, each <= .005',
                         'nominal_family_target': .01, 'deployment_calibrated': False,
                         'limitations': 'Existing fixed-key null qualification failed; this is a frozen nominal target for comparative screening, not verified 1% deployment FPR'},
            'primary_signal_scope': 'Four longer outputs per arm; short-text scores retained separately',
            'review': 'Assistant condition/key-hidden rubric ratings frozen before revealing mapping; not independent human review',
            'candidate_screen': 'Reject promotion on lower long-text matching hits or higher rubric failure count than reference. Ties or apparent improvements only justify a larger fresh study; they cannot establish noninferiority or launch acceptance.',
            'scope': 'Development on four existing keys and four prompts; no quality, detector or launch acceptance'}
    save(public / 'plan.json', plan)
    keys = [(args.prior / f'private/key-{i}').read_bytes() for i in range(4)]
    binding = runtime_binding(max_steps=2048)
    rows = []
    for slot, key in enumerate(keys):
        with Keyprint.from_mlx(args.model, key=key, temperature=.7, top_k=100) as wm:
            reference = wm._candidate
            changed = MixturePublicCandidate(PublicCandidate(temperature=.7, top_k=100))
            if slot == 0:
                save(public / 'identities.json', {'reference': reference.core_identity, 'half_mixture': changed.core_identity})
            for attempt in [a for a in plan['schedule'] if a['key_slot'] == slot]:
                row = dict(attempt, review_id=uuid.uuid4().hex[:12])
                folder = private / row['review_id']
                wm._candidate = changed if row['arm'] == 'half_mixture' else reference
                try:
                    cap = 192 if row['case'] in ('french', 'approval') else 768
                    output = wm.generate(by_case[row['case']]['prompt'], condition='ordinary' if row['arm'] == 'ordinary' else 'marked',
                                         max_tokens=cap, trace=True, output=folder)
                    payload = output.report.get('payload', output.report)
                    if ''.join(t.text for t in output.trace) != output.text:
                        raise AssertionError('Trace does not reconstruct exact output')
                    events = [json.loads(line)['event'] for line in (folder / 'journal.jsonl').read_text().splitlines()]
                    draws = [e for e in events if e['kind'] == 'random_bits_returned']
                    report = json.loads((folder / 'report.json').read_text())
                    row.update(text=output.text, completion=payload['completion'],
                               words=len(output.text.split()), tokens=len(payload['committed_token_ids']),
                               exact_rendering=True, model_forward_calls=report['reservations']['model_forward'],
                               runtime_profile_sha256=output.report['target_identity']['runtime_profile_sha256'],
                               draw_transcript_sha256=hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest())
                    scores = []
                    for key_index in (slot, (slot + 1) % 4):
                        bits = extract(binding, output.text, keys[key_index])
                        score = reference_tail(bits)
                        doubled = reference_tail(bits, double_grid=True)
                        if abs(score['reference_tail'] - doubled['reference_tail']) > 1e-10:
                            raise ArithmeticError('Unstable detector FFT grid')
                        scores.append(score)
                    row.update(scores=scores, matching_hit=scores[0]['reference_tail'] <= .005,
                               other_hit=scores[1]['reference_tail'] <= .005)
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error))
                rows.append(row)
                save(private / 'runs.json', rows)
                print(f"Completed {len(rows)}/48; errors: {sum('error_type' in r for r in rows)}", flush=True)
    reviews = [{'review_id': r['review_id'], 'case': r['case'],
                'prompt': by_case[r['case']]['prompt'], 'rubric': by_case[r['case']]['rubric'],
                'text': r.get('text', ''), 'completion': r.get('completion'),
                'words': r.get('words'), 'error_type': r.get('error_type')} for r in rows]
    random.SystemRandom().shuffle(reviews)
    save(public / 'blind-review.json', reviews)
    print('Condition/key-hidden review ready; mapping and scores remain private.', flush=True)


if __name__ == '__main__':
    main()
