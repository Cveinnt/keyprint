"""Complete-cohort receipt reconciliation; no native replay or quality judgment."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def audit(root, original):
    public, private = root/'public', root/'private'
    read = lambda p: json.loads(p.read_text())
    plan, finished = read(public/'plan.json'), read(public/'results.json')
    identity, rows = read(public/'identity.json'), read(private/'runs.json')
    review = read(private/'blind-review.json')
    if (plan['stage'] != 'study' or not plan['model_load_authorized_in_this_invocation']
            or len(plan['schedule']) != 128 or len(rows) != 128 or finished['attempts'] != 128
            or finished['plan_sha256'] != digest(public/'plan.json')
            or finished['runs_sha256'] != digest(private/'runs.json')
            or finished['identity_sha256'] != digest(public/'identity.json')):
        raise ValueError('Require complete, bound 128-attempt study; a plan or progress is insufficient')
    cases, rubrics = read(private/'cases.json'), read(private/'rubrics.json')
    original_plan = read(original/'public/plan.json')
    if (digest(original/'public/plan.json') != plan['original_plan_sha256']
            or any(digest(original/'private'/name) != original_plan['rubrics_commitment'][name]
                   or read(private/name) != read(original/'private'/name)
                   for name in ('cases.json','rubrics.json'))):
        raise ValueError('Frozen original inputs or rubrics differ')
    from validate_source_grounded import schedule
    if plan['schedule'] != schedule(cases): raise ValueError('Original full schedule differs')
    by_case = {r['id']:r for r in cases}; by_rubric = {r['id']:r for r in rubrics}
    if len(rubrics) != 16 or set(by_rubric) != set(by_case): raise ValueError('Rubrics incomplete')
    ids = [r['review_id'] for r in rows]
    if len(set(ids)) != 128 or any(not re.fullmatch(r'[0-9a-f]{12}', i) for i in ids):
        raise ValueError('Invalid or duplicate review ID')
    if len(review) != 128 or len({r['review_id'] for r in review}) != 128 or {r['review_id'] for r in review} != set(ids):
        raise ValueError('Review packet dropped or duplicated an attempt')
    by_review = {r['review_id']:r for r in review}
    totals = dict(attempts=128, successful_runs=0, audited_runs=0, eos=0, caps=0,
                  errors=0, decode_errors=0, audit_errors=0, committed_tokens=0, audited_tokens=0)
    bindings = []
    for row, scheduled in zip(rows, plan['schedule']):
        if any(row[k] != scheduled[k] for k in ('case','key_slot','condition')):
            raise ValueError('Attempt order or assignment changed')
        folder = private/row['review_id']; result = read(folder/'result.json')
        if (digest(folder/'result.json') != row['result_sha256']
                or digest(folder/'journal.jsonl') != row['journal_sha256']
                or any(row.get(k) != v for k,v in result.items())
                or row['profile_sha256'] != identity['profile_sha256']):
            raise ValueError('Per-output receipt differs')
        visible = by_review[row['review_id']]
        if (set(visible) != {'review_id','case','rubric','text','completion','error_type','decode_error'}
                or visible['case'] != by_case[row['case']] or visible['rubric'] != by_rubric[row['case']]
                or any(visible[k] != row.get(k) for k in ('text','completion','error_type','decode_error'))):
            raise ValueError('Blinded review changed or contains assignment metadata')
        token_count = len(row['committed_token_ids'])
        if not 0 <= token_count <= plan['settings']['max_tokens']:
            raise ValueError('Token count exceeds planned cap')
        totals['committed_tokens'] += token_count
        for metric, test in (
                ('successful_runs', 'error_type' not in row and 'decode_error' not in row),
                ('eos',row['completion']=='eos'), ('caps',row['completion']=='cap'),
                ('errors','error_type' in row), ('decode_errors','decode_error' in row),
                ('audit_errors','audit_error' in row)):
            totals[metric] += test
        if 'audit' in row:
            if ('error_type' in row or 'decode_error' in row or 'audit_error' in row
                    or row['audit']['verified_draws'] != token_count
                    or row['audit']['native_heads_replayed'] is not False
                    or row['audit']['quality_acceptance'] is not False):
                raise ValueError('Invalid per-output audit claim')
            totals['audited_runs'] += 1; totals['audited_tokens'] += token_count
        elif not any(k in row for k in ('error_type','decode_error','audit_error')):
            raise ValueError('Successful output lacks sampler audit')
        scores = row.get('raw_counts')
        if not isinstance(scores,list) or len(scores)!=2: raise ValueError('Missing raw count evidence')
        for score in scores:
            if (score.get('detector_calibrated') is not False or score.get('token_path_only') is not True
                    or any(type(score[k]) is not int or score[k]<0 for k in ('events','ones','trials'))
                    or score['ones']>score['trials'] or score['events']>token_count
                    or score['trials'] != score['events']*30):
                raise ValueError('Invalid uncalibrated token-path counts')
        bindings.append({'review_id':row['review_id'],'result_sha256':row['result_sha256'],
                         'journal_sha256':row['journal_sha256']})
    if any(finished[k] != totals[k] for k in ('successful_runs','audited_runs','eos','caps')):
        raise ValueError('Final summary differs from complete cohort')
    if any(finished[k] is not False for k in ('quality_acceptance','detector_calibrated','launch_ready')):
        raise ValueError('Study metadata cannot assert release acceptance')
    return {'schema':'keyprint.paced-cohort-audit.v1',**totals,
            'plan_sha256':digest(public/'plan.json'),'runs_sha256':digest(private/'runs.json'),
            'results_sha256':digest(public/'results.json'),'identity_sha256':digest(public/'identity.json'),
            'blind_review_sha256':digest(private/'blind-review.json'),
            'cases_sha256':digest(private/'cases.json'),'rubrics_sha256':digest(private/'rubrics.json'),
            'outputs':bindings,'native_heads_replayed':False,'quality_acceptance':False,
            'scope':'Complete retained-cohort/hash reconciliation of in-run sampler audits, not independent native model replay or factual review'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--original',type=Path,required=True)
    args=p.parse_args();result=audit(args.root,args.original)
    with (args.root/'public/cohort-audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('attempts','audited_runs','eos','caps','errors','decode_errors','audit_errors','committed_tokens')}))
