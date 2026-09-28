"""Audit all SDK outputs and join frozen fact-level ratings without losing failures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from multikey_prose_pilot import digest, save


def summarize(plan, rows, ratings):
    signature = lambda row: (row['key_slot'], row['case'], row['condition'])
    expected = {signature(r) for r in plan['schedule']}
    if len(expected) != len(plan['schedule']):
        raise ValueError('Duplicate registered attempts')
    actual = [signature(row) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing, duplicate or undeclared attempts')
    cases = {case['id']: case for case in plan['cases']}
    by_id = {r['review_id']: r for r in rows}
    labels = {r['review_id']: r for r in ratings}
    if len(by_id) != len(rows) or len(labels) != len(ratings) or set(by_id) != set(labels):
        raise ValueError('Missing, duplicate or unknown ratings')
    judgments = {}
    for uid, row in by_id.items():
        label, case = labels[uid], cases[row['case']]
        facts = label['fact_checks']
        if (not isinstance(facts, list) or len(facts) != len(case['facts'])
                or any(type(v) is not bool for v in facts)
                or any(type(label[k]) is not bool for k in ('no_unsupported_claims', 'language_pass', 'prose_pass'))
                or not label.get('reason')):
            raise ValueError('Every fact, language and unsupported-claim check needs a boolean and reason')
        text = row.get('text', '')
        execution = not row.get('error_type') and row.get('completion') == 'eos'
        length = len(text.split()) if 'max_words' in case else len(text)
        unit = 'words' if 'max_words' in case else 'chars'
        length_ok = case['min_' + unit] <= length <= case['max_' + unit]
        paragraphs = len(re.split(r'\n\s*\n', text.strip())) if text.strip() else 0
        format_ok = length_ok and paragraphs == case['paragraphs'] and label['prose_pass']
        content = not row.get('error_type') and bool(text) and all(facts) and label['no_unsupported_claims']
        judgments[uid] = {'length': length, 'paragraphs': paragraphs, 'length_pass': length_ok,
                          'execution_pass': execution, 'content_pass': content,
                          'language_pass': label['language_pass'], 'format_pass': format_ok,
                          'task_pass': execution and content and label['language_pass'] and format_ok,
                          'joint_pass': execution and content and label['language_pass'] and format_ok and row.get('matching_hit', False) and not row.get('other_hit', False)}
    groups = {}
    for condition in ('ordinary', 'marked'):
        selected = [r for r in rows if r['condition'] == condition]
        groups[condition] = {'attempts': len(selected),
            **{name: sum(judgments[r['review_id']][name] for r in selected)
               for name in ('execution_pass', 'content_pass', 'language_pass', 'format_pass', 'task_pass', 'joint_pass')},
            'matching_hits': sum(r.get('matching_hit', False) for r in selected),
            'other_hits': sum(r.get('other_hit', False) for r in selected)}
    paired = {name: Counter() for name in ('content_pass', 'task_pass', 'joint_pass')}
    indexed = {signature(row): row for row in rows}
    for slot, case, condition in expected:
        if condition != 'ordinary':
            continue
        a = judgments[indexed[(slot, case, 'ordinary')]['review_id']]
        b = judgments[indexed[(slot, case, 'marked')]['review_id']]
        for name in paired:
            x, y = a[name], b[name]
            paired[name]['both_pass' if x and y else 'ordinary_only' if x else 'marked_only' if y else 'both_fail'] += 1
    breakdown = []
    for case in cases:
        for condition in groups:
            subset = [r for r in rows if r['case'] == case and r['condition'] == condition]
            breakdown.append({'case': case, 'condition': condition, 'attempts': len(subset),
                              **{name: sum(judgments[r['review_id']][name] for r in subset)
                                 for name in ('content_pass', 'language_pass', 'format_pass', 'task_pass', 'joint_pass')}})
    return {'groups': groups, 'paired': {k: dict(v) for k, v in paired.items()},
            'by_case': breakdown, 'judgments': judgments,
            'quality_acceptance': False, 'detector_calibrated': False,
            'scope': plan['scope'],
            'interpretation': 'Joint pass requires factual, language, length, paragraph/prose and execution pass plus matching-key hit and no other-key hit on the same output. Descriptive uncalibrated development measure, never general quality or deployment acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    public = args.root / 'public'
    plan = json.loads((public / 'plan.json').read_text())
    commitment = json.loads((public / 'rating-commitment.json').read_text())
    for name in ('plan.json', 'blind-review.json', 'frozen-ratings.json'):
        if digest(public / name) != commitment[name]:
            raise ValueError('Review commitment changed')
    for name, expected in {'validate_long_fidelity.py': plan['script_sha256'],
                           'long_fidelity_cases.json': plan['cases_sha256'],
                           **plan['source_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Registered source changed')
    rows = json.loads((args.root / 'private/runs.json').read_text())
    ratings = json.loads((public / 'frozen-ratings.json').read_text())
    summary = summarize(plan, rows, ratings)
    from audit_pydantic_ai import audit_journal
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding = runtime_binding(max_steps=2048)
    audited = []
    for row in rows:
        if row.get('error_type'):
            continue
        folder = args.root / 'private' / row['review_id']
        report = json.loads((folder / 'report.json').read_text())
        data, payload = report['report'], report['report']['payload']
        ids = payload['committed_token_ids']
        raw = b''.join(binding.token_bytes[i] or b'' for i in ids)
        visible = [c for c in data['carrier_rendering'] if c['channel'] == 'visible']
        if len(visible) != 1:
            raise ValueError('Expected one visible carrier')
        pending = bytes.fromhex(visible[0].get('pending_utf8_hex', ''))
        if (raw != row['text'].encode('utf-8') + pending
                or data['rendered_carriers']['visible_text'] != row['text']
                or len(ids) != row['tokens'] or payload['assigned_condition'] != row['condition']):
            raise ValueError('Native bytes, rendered text, token count or assigned condition differ')
        events = audit_journal(folder / 'journal.jsonl')
        if (sum(e['kind'] == 'committed_step' for e in events) != len(ids)
                or report['reservations']['model_forward'] <= 0):
            raise ValueError('Missing model or committed-token evidence')
        draws = [e for e in events if e['kind'] == 'random_bits_returned']
        if not draws:
            raise ValueError('Missing actual draws')
        audited.append({'review_id': row['review_id'], 'tokens': len(ids),
                        'report_sha256': digest(folder / 'report.json'),
                        'journal_sha256': digest(folder / 'journal.jsonl'),
                        'draw_transcript_sha256': hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()})
    result = {'summary': summary, 'runs': rows,
              'summary_script_sha256': digest(Path(__file__)), 'plan_sha256': digest(public / 'plan.json'),
              'freshness': {'attempts': len(audited), 'tokens': sum(r['tokens'] for r in audited),
                            'distinct_draw_transcripts': len({r['draw_transcript_sha256'] for r in audited}),
                            'rows': audited}}
    save(public / 'results.json', result)
    print(json.dumps({'groups': summary['groups'], 'paired': summary['paired'],
                      'audited_attempts': len(audited), 'tokens': result['freshness']['tokens']}, indent=2))


if __name__ == '__main__':
    main()
