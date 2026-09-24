"""Join committed ratings only after condition/key-hidden review is complete."""
import argparse
import hashlib
import json
from pathlib import Path

from multikey_prose_pilot import digest, repetition_summary, save
from prose_quality_pilot import summarize


def audit(root):
    public = root / 'public'
    plan = json.loads((public / 'plan.json').read_text())
    commitment = json.loads((public / 'rating-commitment.json').read_text())
    for filename, field in [('plan.json', 'plan_sha256'),
                            ('blind-review.json', 'blind_review_sha256'),
                            ('frozen-ratings.json', 'ratings_sha256')]:
        if digest(public / filename) != commitment[field]:
            raise ValueError('Frozen review commitment does not match')
    if digest(Path(__file__).with_name('multikey_prose_pilot.py')) != plan['script_sha256']:
        raise ValueError('Generation script changed after plan registration')
    if digest(Path(__file__).with_name('prose_quality_cases.json')) != plan['cases_sha256']:
        raise ValueError('Registered cases changed')
    rows = json.loads((root / 'private/runs.json').read_text())
    ratings = json.loads((public / 'frozen-ratings.json').read_text())
    if len(ratings) != commitment['review_count']:
        raise ValueError('Frozen review count does not match')
    repetition = repetition_summary(plan, rows)
    # Flatten registered key/repetition pairs without treating them as new prompts.
    flat = [dict(r, repetition=r['key_slot'] * plan['repeats'] + r['repetition']) for r in rows]
    summary = summarize(dict(plan, repeats=plan['keys'] * plan['repeats']), flat, ratings)
    summary['scope'] = 'Three reused prompts, four fresh keys; descriptive assistant review, not independent human acceptance or a powered quality study.'
    labels = {r['review_id']: r for r in ratings}
    breakdown = []
    for slot in range(plan['keys']):
        for case in plan['cases']:
            for condition in ('ordinary', 'marked'):
                group = [r for r in rows if (r['key_slot'], r['case'], r['condition']) == (slot, case['id'], condition)]
                breakdown.append({'key_slot': slot, 'case': case['id'], 'condition': condition,
                                  'attempts': len(group), 'task_pass': sum(
                                      labels[r['review_id']]['task_pass'] and r.get('completion') == 'eos'
                                      and not r.get('error_type') for r in group)})
    freshness = []
    for row in rows:
        if row.get('error_type'):
            continue
        folder = root / 'private' / row['review_id']
        report = json.loads((folder / 'report.json').read_text())
        events = [json.loads(line)['event'] for line in (folder / 'journal.jsonl').read_text().splitlines()]
        draws = [e for e in events if e['kind'] == 'random_bits_returned']
        payload = report['report']['payload']
        freshness.append({'review_id': row['review_id'], 'draw_calls': len(draws),
                          'draw_transcript_sha256': hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest(),
                          'model_forward_calls': report['reservations']['model_forward'],
                          'tokens_match': len(payload['committed_token_ids']) == row['tokens']})
    result = {'summary_script_sha256': digest(Path(__file__)),
              'plan_sha256': digest(public / 'plan.json'),
              'frozen_ratings_sha256': digest(public / 'frozen-ratings.json'),
              'summary': summary, 'breakdown': breakdown, 'repetition': repetition,
              'language_pass': sum(r['language_pass'] for r in ratings),
              'freshness': {'attempts_audited': len(freshness),
                            'unique_draw_transcripts': len({r['draw_transcript_sha256'] for r in freshness}),
                            'all_model_calls_recorded': all(r['model_forward_calls'] > 0 for r in freshness),
                            'all_draws_recorded': all(r['draw_calls'] > 0 for r in freshness),
                            'all_token_counts_match': all(r['tokens_match'] for r in freshness),
                            'scope': 'Fresh calls and distinct transcripts do not establish statistical RNG quality.',
                            'rows': freshness}, 'runs': rows}
    save(public / 'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    result = audit(parser.parse_args().root)
    print(json.dumps({k: result[k] for k in ('summary', 'breakdown', 'language_pass')}, indent=2))
