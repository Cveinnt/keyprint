"""Recount frozen semantic reviews without models, rerating or changing evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def recount(plan, rows, ratings, blind):
    signature = lambda r: (r['case'], r['key_slot'], r['condition'])
    expected = [signature(r) for r in plan['schedule']]
    actual = [signature(r) for r in rows]
    require(len(set(expected)) == len(expected) and Counter(actual) == Counter(expected),
            'Missing, duplicate or undeclared attempt')
    ids = [r['review_id'] for r in rows]
    require(len(set(ids)) == len(ids), 'Duplicate output identity')
    for collection in (ratings, blind):
        require(Counter(r['review_id'] for r in collection) == Counter(ids),
                'Missing, duplicate or unknown review')
    labels = {r['review_id']: r for r in ratings}
    views = {r['review_id']: r for r in blind}
    groups = {c: Counter() for c in ('ordinary', 'marked')}
    outcomes = {}
    for row in rows:
        label, view = labels[row['review_id']], views[row['review_id']]
        require(all(view.get(k) == row.get(k) for k in ('text', 'completion', 'error_type')),
                'Blinded output differs from original')
        require(view['case']['id'] == row['case'], 'Review source identity changed')
        facts = view.get('rubric', {}).get('essential_facts', view['case'].get('facts'))
        flags = label['fact_checks']
        require(isinstance(flags, list) and len(flags) == len(facts) and len(flags) > 0,
                'Incomplete fact review')
        require(all(type(v) is bool for v in flags +
                    [label['no_unsupported_claims'], label['language_pass']]),
                'Invalid semantic rating')
        require(bool(label.get('reason', '').strip()), 'Missing review reason')
        content = (bool(row.get('text')) and not row.get('error_type') and
                   all(flags) and label['no_unsupported_claims'])
        groups[row['condition']].update(attempts=1, content_pass=int(content),
            language_pass=int(label['language_pass']),
            complete=int(row.get('completion') == 'eos' and not row.get('error_type')))
        outcomes[signature(row)] = content
    paired = Counter()
    for case, key, condition in expected:
        if condition != 'ordinary':
            continue
        a, b = outcomes[case, key, 'ordinary'], outcomes[case, key, 'marked']
        paired['both_pass' if a and b else 'ordinary_only' if a else
               'marked_only' if b else 'both_fail'] += 1
    return {'groups': {c: dict(v) for c, v in groups.items()}, 'paired_content': dict(paired)}


def audit(root, published, source_study=False):
    public, private = root / 'public', root / 'private'
    commitment = read(published / 'rating-commitment.json')
    review_dir = private if source_study else public
    paths = {'plan.json': public / 'plan.json',
             'blind-review.json': review_dir / 'blind-review.json',
             'frozen-ratings.json': review_dir / 'frozen-ratings.json'}
    for name, path in paths.items():
        require(digest(path) == commitment[name], 'Frozen review changed: ' + name)
    rows = read(private / 'runs.json')
    result = recount(read(paths['plan.json']), rows, read(paths['frozen-ratings.json']),
                     read(paths['blind-review.json']))
    saved = read(published / ('fidelity-results.json' if source_study else 'results.json'))
    if source_study:
        receipt = read(published / 'receipt-audit.json')
        require(receipt['runs_sha256'] == digest(private / 'runs.json'), 'Audited runs changed')
        old = saved['strict']['groups']
        metric = 'content'
    else:
        require(rows == saved['runs'], 'Published original runs changed')
        old = saved['summary']['groups']
        metric = 'content_pass'
    for c, values in result['groups'].items():
        require(values['attempts'] == old[c]['attempts'] and
                values['content_pass'] == old[c][metric] and
                values['language_pass'] == old[c]['language' if source_study else 'language_pass'],
                'Recomputed totals differ from retained result')
    result.update(input_hashes={name: digest(path) for name, path in paths.items()},
                  runs_sha256=digest(private / 'runs.json'),
                  published_result_sha256=digest(published / ('fidelity-results.json' if source_study else 'results.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fact-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parents[1] / 'evidence'
    result = {
        'schema': 'keyprint.semantic-evidence-recheck.v1',
        'multilingual_released_sdk': audit(args.fact_root, evidence / 'fact-fidelity-2026-09-28'),
        'dolly_experimental_qwen35': audit(args.source_root, evidence / 'source-grounded-run-2026-09-28', True),
        'quality_acceptance': False,
        'scope': 'Recounts existing frozen assistant judgments and verifies their commitments and output joins. No new inference, independent semantic rerating, human approval, causal estimate or pooled pass rate. Language retention is distinct from fact preservation.',
        'script_sha256': digest(Path(__file__)),
    }
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for name in ('multilingual_released_sdk', 'dolly_experimental_qwen35'):
        print(name, result[name]['groups'], result[name]['paired_content'])


if __name__ == '__main__':
    main()
