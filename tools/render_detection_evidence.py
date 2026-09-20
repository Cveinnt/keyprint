"""Build an offline, interactive view of audited prompt-aware evidence.

Source answers and secret keys never enter the exported human-control records.
An explicit partial audit produces a visibly incomplete snapshot, never a pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return json.loads(path.read_text())


def bind(path, digest):
    if sha(path.read_bytes()) != digest:
        raise ValueError(f'Audited file changed: {path.name}')


def script_data(value):
    # External script plus escaped HTML metacharacters; no source text becomes
    # executable markup, and U+2028/U+2029 stay portable across JS parsers.
    data = json.dumps(value, ensure_ascii=True, allow_nan=False)
    return 'window.KEYPRINT_EVIDENCE = ' + data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026') + ';\n'


def null_rows(study, partial_audit=None):
    public = study / 'public'; plan = load(public / 'plan.json')
    audit = load(partial_audit or public / 'integrity.json')
    expected = 'partial_integrity_pass' if partial_audit else 'pass'
    if audit['status'] != expected:
        raise ValueError('Expected completed integrity audit or explicit partial snapshot')
    bind(public / 'plan.json', audit['plan_sha256'])
    raw = (public / 'results.jsonl').read_bytes()
    prefix = raw[:audit['results_prefix_bytes']]
    if not prefix.endswith(b'\n') or sha(prefix) != audit['results_prefix_sha256']:
        raise ValueError('Audited result prefix changed')
    rows = [json.loads(line) for line in prefix.splitlines()]
    if len(rows) != audit['totals']['records'] or len(rows) > plan['planned_documents']:
        raise ValueError('Audited result count differs')
    available = [r for r in rows if 'error' not in r]
    if len(available) != audit['totals']['available']:
        raise ValueError('Audited availability differs')
    summary = None
    if not partial_audit:
        if prefix != raw:
            raise ValueError('Final audit does not bind all results')
        bind(public / 'summary.json', audit['summary_sha256'])
        summary = load(public / 'summary.json')
        if summary['available'] != len(available) or summary['attempts'] != len(rows):
            raise ValueError('Summary count differs')
        if summary['false_hits'] != sum(r['flagged'] for r in available):
            raise ValueError('Summary flag count differs')
        if summary['status'] == 'completed':
            bound = summary['iid_only_upper_97_5_percent']
            if (len(available) != plan['planned_documents'] or summary['fatal'] is not None
                    or type(bound) not in (int, float) or not math.isfinite(bound) or not 0 <= bound <= 1
                    or summary['null_screen_passed'] != (bound <= .01)):
                raise ValueError('Completed summary requires all controls and a valid bound')
    for index, row in enumerate(rows):
        if row['id'] != f'null-{index:03d}' or row['source_index'] != plan['tasks'][index]['source_index']:
            raise ValueError('Null task order differs')
        if 'error' not in row and (len(row['working_log_ratios']) != 2
                or row['flagged'] != any(s >= plan['fixed_log_cutoff'] for s in row['working_log_ratios'])):
            raise ValueError('Null decision differs')
    controls = [{k: r[k] for k in ('id', 'category', 'source_index', 'words')} | (
        {'error': r['error']['type']} if 'error' in r else
        {'scores': r['working_log_ratios'], 'flagged': r['flagged']}) for r in rows]
    return plan, summary, controls, audit


def build(confirmation, null_study, *, partial_audit=None):
    public = confirmation / 'public'; plan = load(public / 'plan.json')
    audit = load(public / 'integrity.json'); summary = load(public / 'summary.json')
    if audit['status'] != 'pass' or summary['status'] != 'completed':
        raise ValueError('Completed audited paired confirmation required')
    bind(public / 'plan.json', audit['plan_sha256']); bind(public / 'summary.json', audit['summary_sha256'])
    null_plan, null_summary, controls, null_audit = null_rows(null_study, partial_audit)
    if (null_plan['key_commitments'] != plan['key_commitments']
            or null_plan['fixed_log_cutoff'] != plan['fixed_log_cutoff']
            or plan['fixed_log_cutoff'] != math.log(200.)):
        raise ValueError('Paired and null key/cutoff identities differ')
    for name, digest in null_plan['confirmation_sha256'].items():
        bind(public / name, digest)
    cases = []
    for index, task in enumerate(plan['tasks']):
        case = {k: task[k] for k in ('source_index', 'category', 'prompt')}
        for condition in ('ordinary', 'marked'):
            name = f'confirmation-{index:02d}-{condition}'; row = load(public / (name + '.json'))
            if row['id'] != name or row['source_index'] != task['source_index'] or 'error' in row:
                raise ValueError('Paired task identity differs')
            bind(confirmation / (name + '.heads.json'), row['heads_sha256'])
            bind(confirmation / (name + '.scores.json'), row['scores_sha256'])
            heads = load(confirmation / (name + '.heads.json')); scores = load(confirmation / (name + '.scores.json'))
            if (heads['text_sha256'] != sha(row['text'].encode())
                    or heads['original_prompt_sha256'] != task['prompt_sha256']
                    or row['working_log_ratios'] != [s['working_log_ratio'] for s in scores]
                    or row['matching_flagged'] != scores[row['key_index']]['flagged']
                    or row['other_flagged'] != scores[1-row['key_index']]['flagged']):
                raise ValueError('Paired text or score differs from retained receipts')
            case[condition] = {'text': row['text'], 'words': row['words'], 'completion': row['completion'],
                'key_index': row['key_index'], 'scores': row['working_log_ratios'],
                'matching_flagged': row['matching_flagged'], 'other_flagged': row['other_flagged']}
        cases.append(case)
    if len(cases) != 12:
        raise ValueError('All twelve paired tasks required')
    return {'title': 'Keyprint · A signal, and its controls', 'partial': bool(partial_audit),
        'cutoff': plan['fixed_log_cutoff'], 'cases': cases, 'controls': controls,
        'confirmation': summary, 'null_summary': null_summary,
        'planned_controls': null_plan['planned_documents'],
        'source': plan['source'], 'license': plan['license'], 'attribution': plan['attribution'],
        'provenance': {'confirmation_plan_sha256': audit['plan_sha256'],
            'null_plan_sha256': null_audit['plan_sha256'],
            'null_results_sha256': null_audit['results_prefix_sha256'],
            'model': 'Qwen3-8B · pinned MLX 4-bit', 'score': 'Original-prompt half-mixture likelihood',
            'kernels_independently_rerun': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('confirmation', 'null-study', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--partial-audit', type=Path)
    args = parser.parse_args()
    data = build(args.confirmation, args.null_study, partial_audit=args.partial_audit)
    args.output.mkdir()
    assets = Path(__file__).with_name('evidence_assets')
    for path in assets.iterdir():
        if path.is_file(): shutil.copyfile(path, args.output / path.name)
    shutil.copyfile(Path(__file__).parents[1] / 'src/keyprint/web/reader.js', args.output / 'reader.js')
    (args.output / 'data.js').write_text(script_data(data))
    (args.output / 'data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({'path': str((args.output / 'index.html').resolve()), 'pairs': len(data['cases']),
        'controls': len(data['controls']), 'partial': data['partial']}))


if __name__ == '__main__': main()
