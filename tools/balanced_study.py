"""Complete receipt reconciliation for the frozen balanced development study.

Checks retained in-run audits, not an independent native model-head replay.
Never turns a plan, partial run or loaded profile identity into acceptance.
"""
import json
from pathlib import Path
import re

from audit_paced_study import digest
from freeze_balanced_evaluation import validate


def read(path):return json.loads(path.read_text())


def validate_preflight(root, supervisor):
    required=('public/plan.json','public/results.json','public/identity.json','private/runs.json')
    if any(not (root/name).is_file() for name in required):
        raise ValueError('Completed balanced native preflight required; partial artifacts are insufficient')
    plan,result,identity,rows=(read(root/name) for name in required)
    guard=read(supervisor/'result.json')
    if guard['status']!='completed' or guard['exit_code']!=0 or guard['cleanup_verified'] is not True:
        raise ValueError('Preflight supervisor did not complete with verified cleanup')
    from preflight_balanced import CASES
    from balanced_source_session import policy_spec
    helpers={'preflight_balanced.py','balanced_inference.py','balanced_source_session.py',
             'balanced_score.py','centered_score.py','paced_source_session.py',
             'paced_integer_kernel.py','replay_wide_mlx.py','memory_watchdog.py','mlx_guarded_worker.py'}
    expected=[dict(case=c['id'],key_slot=i,condition=arm) for i,c in enumerate(CASES)
              for arm in ('ordinary','marked')]
    if (plan['schema']!='keyprint.balanced-native-preflight-plan.v1' or plan['stage']!='preflight'
            or plan['schedule']!=expected or plan['policy']!=policy_spec()
            or set(plan['scripts_sha256'])!=helpers
            or plan['settings']!=dict(max_tokens=16,temperature=.7,top_k=100,profile_max_steps=1024,enable_thinking=False)
            or result['schema']!='keyprint.balanced-native-preflight-results.v1'
            or result['attempts']!=4 or result['failures']!=0 or result['audited_runs']!=4 or len(rows)!=4
            or result['plan_sha256']!=digest(root/'public/plan.json')
            or result['runs_sha256']!=digest(root/'private/runs.json')
            or result['identity_sha256']!=digest(root/'public/identity.json')):
        raise ValueError('Preflight scope or completion differs')
    for name,h in plan['scripts_sha256'].items():
        if digest(Path(__file__).with_name(name))!=h:raise ValueError('Preflight helper changed')
    if (read(root/'private/cases.json')!=list(CASES)
            or digest(root/'private/cases.json')!=plan['cases_sha256']):
        raise ValueError('Preflight material changed')
    import hashlib
    keys=[(root/f'private/key-{i}').read_bytes() for i in range(2)]
    if any(len(k)!=32 for k in keys) or [hashlib.sha256(k).hexdigest() for k in keys]!=plan['key_sha256']:
        raise ValueError('Preflight keys changed')
    seen=set()
    for row,scheduled in zip(rows,expected,strict=True):
        if (not re.fullmatch(r'[0-9a-f]{12}',row['review_id']) or row['review_id'] in seen
                or any(row[k]!=v for k,v in scheduled.items())):raise ValueError('Preflight row differs')
        seen.add(row['review_id']);folder=root/'private'/row['review_id']
        saved=read(folder/'result.json')
        if (digest(folder/'result.json')!=row['result_sha256']
                or digest(folder/'journal.jsonl')!=row['journal_sha256']
                or any(row.get(k)!=v for k,v in saved.items())
                or any(k in row for k in ('error_type','decode_error','audit_error'))
                or row['profile_sha256']!=identity['profile_sha256']
                or row['completion'] not in ('eos','cap')
                or not 1<=len(row['committed_token_ids'])<=16
                or (row['completion']=='cap' and len(row['committed_token_ids'])!=16)
                or row['audit']!=dict(verified_draws=len(row['committed_token_ids']),native_heads_replayed=False,quality_acceptance=False)):
            raise ValueError('Preflight retained output differs')
    if (result['committed_tokens']!=sum(len(r['committed_token_ids']) for r in rows)
            or result['caps']!=sum(r['completion']=='cap' for r in rows)
            or result['eos']!=sum(r['completion']=='eos' for r in rows)
            or any(result[k] is not False for k in ('quality_acceptance','detector_calibrated','launch_ready'))):
        raise ValueError('Preflight totals or acceptance claim differs')
    return dict(plan_sha256=digest(root/'public/plan.json'),results_sha256=digest(root/'public/results.json'),
                supervisor_sha256=digest(supervisor/'result.json'),profile_sha256=identity['profile_sha256'],
                model_assets=plan['model_assets'],dependencies=plan['dependencies'],sdk_source_sha256=plan['sdk_source_sha256'])


def blind_packet(rows,cases,rubrics):
    by_case={r['id']:r for r in cases};by_rubric={r['id']:r for r in rubrics}
    return [dict(review_id=r['review_id'],case=by_case[r['case']],rubric=by_rubric[r['case']],
                 text=r.get('text'),completion=r['completion'],error_type=r.get('error_type') or
                 ('SamplerAuditError' if r.get('audit_error') else None),
                 decode_error=r.get('decode_error')) for r in rows]


def totals(rows):
    return dict(attempts=len(rows),committed_tokens=sum(len(r['committed_token_ids']) for r in rows),
                successful_runs=sum(not any(k in r for k in ('error_type','decode_error','audit_error')) for r in rows),
                audited_runs=sum('audit' in r for r in rows),
                eos=sum(r['completion']=='eos' for r in rows),caps=sum(r['completion']=='cap' for r in rows),
                errors=sum('error_type' in r for r in rows),decode_errors=sum('decode_error' in r for r in rows),
                audit_errors=sum('audit_error' in r for r in rows))


def audit(root,bundle):
    manifest=validate(bundle)
    plan=read(root/'public/plan.json');finished=read(root/'public/results.json')
    identity=read(root/'public/identity.json');rows=read(root/'private/runs.json')
    review=read(root/'private/blind-review.json')
    if (plan['schema']!='keyprint.balanced-evaluation-run.v1' or plan['stage']!='study'
            or plan['model_load_authorized_in_this_invocation'] is not True
            or plan['bundle_manifest_sha256']!=digest(bundle/'public/manifest.json')
            or plan['bundle_protocol_sha256']!=digest(bundle/'public/protocol.json')
            or not plan.get('preflight')
            or plan['preflight']['profile_sha256']!=identity['profile_sha256']
            or plan['schedule']!=manifest['schedule'] or plan['policy']!=manifest['policy']
            or plan['settings']!=read(bundle/'public/protocol.json')['settings']
            or len(rows)!=128 or len({r['review_id'] for r in rows})!=128
            or finished['plan_sha256']!=digest(root/'public/plan.json')
            or finished['runs_sha256']!=digest(root/'private/runs.json')
            or finished['identity_sha256']!=digest(root/'public/identity.json')
            or finished['review_sha256']!=digest(root/'private/blind-review.json')):
        raise ValueError('Complete bound 128-attempt balanced study required')
    if (finished['schema']!='keyprint.balanced-evaluation-results.v1'
            or any(document[k] is not False for document in (plan,finished)
                   for k in ('quality_acceptance','detector_calibrated','launch_ready'))):
        raise ValueError('Execution receipts cannot claim acceptance')
    cases=read(root/'private/cases.json');rubrics=read(root/'private/rubrics.json')
    if (cases!=read(bundle/'private/cases.json') or rubrics!=read(bundle/'private/rubrics.json')
            or len(review)!=128 or len({r['review_id'] for r in review})!=128):
        raise ValueError('Review inputs changed or incomplete')
    expected={r['review_id']:r for r in blind_packet(rows,cases,rubrics)}
    if any(r!=expected.get(r['review_id']) for r in review):raise ValueError('Blind review text or metadata differs')
    for row,scheduled in zip(rows,manifest['schedule'],strict=True):
        if not re.fullmatch(r'[0-9a-f]{12}',row['review_id']) or any(row[k]!=v for k,v in scheduled.items()):
            raise ValueError('Attempt identity, assignment or order differs')
        folder=root/'private'/row['review_id'];saved=read(folder/'result.json')
        if (digest(folder/'result.json')!=row['result_sha256']
                or digest(folder/'journal.jsonl')!=row['journal_sha256']
                or any(row.get(k)!=v for k,v in saved.items())
                or row['profile_sha256']!=identity['profile_sha256']
                or row['completion'] not in ('eos','cap','incomplete')
                or len(row['committed_token_ids'])>plan['settings']['max_tokens']
                or row['quality_acceptance'] is not False or row['detector_calibrated'] is not False):
            raise ValueError('Retained output receipt differs')
        failed=any(k in row for k in ('error_type','decode_error','audit_error'))
        if not failed:
            if (row.get('audit')!=dict(verified_draws=len(row['committed_token_ids']),native_heads_replayed=False,quality_acceptance=False)
                    or not row['committed_token_ids'] or row['completion'] not in ('eos','cap')
                    or (row['completion']=='cap' and len(row['committed_token_ids'])!=plan['settings']['max_tokens'])):
                raise ValueError('Successful output lacks valid in-run audit')
        scores=row['raw_counts']
        if len(scores)!=2:raise ValueError('Owner and next-key scores required')
        for score in scores:
            if failed or row['completion']!='eos':
                if 'unavailable' not in score:raise ValueError('Failed or capped response cannot be a clean score control')
            elif (score.get('token_path_only') is not True or score.get('detector_calibrated') is not False
                  or score.get('profile_sha256')!=identity['profile_sha256']
                  or any(type(score[k]) is not int or score[k]<0 for k in ('ones','events','trials'))
                  or score['ones']>score['trials'] or score['events']>len(row['committed_token_ids'])
                  or score['trials']!=score['events']*30):
                raise ValueError('Invalid token-path score')
    summary=totals(rows)
    if any(finished[k]!=v for k,v in summary.items()):raise ValueError('Complete-cohort totals differ')
    return dict(**summary,plan_sha256=digest(root/'public/plan.json'),runs_sha256=digest(root/'private/runs.json'),
                results_sha256=digest(root/'public/results.json'),blind_review_sha256=digest(root/'private/blind-review.json'),
                cases_sha256=digest(root/'private/cases.json'),rubrics_sha256=digest(root/'private/rubrics.json'),
                bundle_manifest_sha256=digest(bundle/'public/manifest.json'),
                native_heads_replayed=False,quality_acceptance=False,
                scope='Complete receipt reconciliation of retained in-run sampler audits; not independent model replay or quality acceptance')
