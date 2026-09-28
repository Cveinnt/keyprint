"""Join frozen, condition-hidden ratings with every planned mixture attempt."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ARMS = ('ordinary', 'reference', 'half_mixture')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(plan, rows, labels):
    def signature(row):
        return tuple(row[k] for k in ('key_slot', 'case', 'repetition', 'arm'))
    expected = {signature(r) for r in plan['schedule']}
    actual = [signature(r) for r in rows]
    ids = {r['review_id'] for r in rows}
    if set(actual) != expected or len(actual) != len(expected) or len(ids) != len(rows):
        raise ValueError('Missing, duplicate or undeclared attempts')
    ratings = {}
    for label in labels:
        if label['review_id'] in ratings or label['review_id'] not in ids:
            raise ValueError('Unknown or duplicate review')
        if any(type(label[k]) is not bool for k in ('task_pass', 'language_pass')) or not label.get('reason'):
            raise ValueError('Each review needs booleans and an explicit reason')
        ratings[label['review_id']] = label
    if set(ratings) != ids:
        raise ValueError('Every retained attempt must be reviewed')
    outcomes = {r['review_id']: bool(ratings[r['review_id']]['task_pass'] and
                                   not r.get('error_type') and r.get('completion') == 'eos') for r in rows}
    groups = {}
    by_case = []
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        longer = [r for r in selected if r['case'] not in ('french', 'approval')]
        groups[arm] = {'attempts': len(selected), 'task_pass': sum(outcomes[r['review_id']] for r in selected),
                       'errors': sum('error_type' in r for r in selected),
                       'language_pass': sum(ratings[r['review_id']]['language_pass'] for r in selected),
                       'non_eos': sum(r.get('completion') != 'eos' for r in selected),
                       'long_detection': {'planned': len(longer),
                           'available': sum('scores' in r and not r.get('error_type') for r in longer),
                           'matching_hits': sum(r.get('matching_hit', False) for r in longer),
                           'other_hits': sum(r.get('other_hit', False) for r in longer)}}
        for case in plan['cases']:
            group = [r for r in selected if r['case'] == case['id']]
            by_case.append({'arm': arm, 'case': case['id'], 'attempts': len(group),
                            'task_pass': sum(outcomes[r['review_id']] for r in group),
                            'matching_hits': sum(r.get('matching_hit', False) for r in group),
                            'other_hits': sum(r.get('other_hit', False) for r in group)})
    repetition = []
    for key in plan['keys']:
        for arm in ARMS:
            group = [r for r in rows if r['key_slot'] == key and r['case'] == 'french' and r['arm'] == arm]
            texts = [r['text'] for r in group if not r.get('error_type') and r.get('completion') == 'eos']
            repetition.append({'key_slot': key, 'arm': arm, 'attempts': len(group),
                               'completed': len(texts), 'distinct_texts': len(set(texts))})
    paired = Counter()
    indexed = {signature(r): r for r in rows}
    for row in rows:
        if row['arm'] != 'reference':
            continue
        other = indexed[(row['key_slot'], row['case'], row['repetition'], 'half_mixture')]
        a, b = outcomes[row['review_id']], outcomes[other['review_id']]
        paired['both_pass' if a and b else 'reference_only' if a else 'mixture_only' if b else 'both_fail'] += 1
    ref, mix = groups['reference'], groups['half_mixture']
    complete = all(g['errors'] == 0 and g['long_detection']['available'] == g['long_detection']['planned']
                   for g in groups.values())
    rejection = []
    if not complete:
        rejection.append('incomplete inference or scoring')
    if mix['task_pass'] < ref['task_pass']:
        rejection.append('fewer rubric passes than reference')
    if mix['long_detection']['matching_hits'] < ref['long_detection']['matching_hits']:
        rejection.append('fewer long-text matching hits at frozen detector rule')
    return {'complete': complete, 'groups': groups, 'by_case': by_case, 'repetition': repetition,
            'paired_reference_mixture': dict(paired), 'candidate_rejection_reasons': rejection,
            'development_screen': 'rejected' if rejection else 'larger_fresh_study_required',
            'sdk_promotion': False, 'quality_acceptance': False, 'detector_calibrated': False,
            'scope': plan['scope'], 'detector_limitations': plan['detector']['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    public = args.root / 'public'
    plan = json.loads((public / 'plan.json').read_text())
    commitment = json.loads((public / 'rating-commitment.json').read_text())
    for filename in ('plan.json', 'blind-review.json', 'frozen-ratings.json'):
        if sha(public / filename) != commitment[filename]:
            raise ValueError('Frozen review commitment changed')
    if sha(Path(__file__).with_name('compare_mixture_candidate.py')) != plan['script_sha256']:
        raise ValueError('Registered generation script changed')
    for name, expected in plan['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Registered experiment source changed')
    rows = json.loads((args.root / 'private/runs.json').read_text())
    ratings = json.loads((public / 'frozen-ratings.json').read_text())
    result = summarize(plan, rows, ratings)
    audited = []
    identities = json.loads((public / 'identities.json').read_text())
    for row in rows:
        if row.get('error_type'):
            continue
        folder = args.root / 'private' / row['review_id']
        report = json.loads((folder / 'report.json').read_text())
        receipt = report['receipt']
        commits = receipt['committed_token_ids']
        events = [json.loads(line)['event'] for line in (folder / 'journal.jsonl').read_text().splitlines()]
        journal_ids = [e['token_id'] for e in events if e['kind'] == 'committed_step']
        draws = [e for e in events if e['kind'] == 'random_bits_returned']
        expected_identity = identities['half_mixture' if row['arm'] == 'half_mixture' else 'reference']
        if report['runtime'] != expected_identity or row['runtime_profile_sha256'] != expected_identity['runtime_profile_sha256']:
            raise ValueError('Research runtime identity differs')
        if row['text'] != report['text'] or commits != journal_ids or len(commits) != row['tokens']:
            raise ValueError('Text or committed-token receipt differs')
        if hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest() != row['draw_transcript_sha256']:
            raise ValueError('Random-draw transcript differs')
        if sum(e['kind'] == 'model_forward_requested' for e in events) != row['model_forward_calls']:
            raise ValueError('Model-forward count differs')
        audited.append({'review_id': row['review_id'], 'tokens': len(commits),
                        'report_sha256': sha(folder / 'report.json'), 'journal_sha256': sha(folder / 'journal.jsonl')})
    result.update(summary_script_sha256=sha(Path(__file__)), plan_sha256=sha(public / 'plan.json'),
                  ratings_sha256=sha(public / 'frozen-ratings.json'),
                  freshness={'audited_attempts': len(audited), 'audited_tokens': sum(r['tokens'] for r in audited),
                             'unique_draw_transcripts': len({r['draw_transcript_sha256'] for r in rows if not r.get('error_type')}),
                             'rows': audited})
    (public / 'results.json').write_text(json.dumps({'summary': result, 'runs': rows}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('groups', 'paired_reference_mixture', 'development_screen', 'candidate_rejection_reasons')}, indent=2))


if __name__ == '__main__':
    main()
