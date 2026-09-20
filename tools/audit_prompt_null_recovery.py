"""Verify operational recovery provenance after the full numerical null audit.

The ordinary audit must first reconcile all combined records and every term.
This extra audit binds the failed parent, unchanged retained records, precise
unmeasured recovery set and cache-disabled preflight. Original failure persists.
"""
import argparse
import json
from pathlib import Path

from recover_prompt_null import parent_state, load, bind
from prompt_confirmation_selection import sha


def check_record(original, recovered):
    if 'error' not in original:
        if recovered != {**original,'execution_origin':'retained'}:
            raise ValueError('Retained parent record changed')
    elif (recovered.get('execution_origin')!='recovery'
          or recovered.get('original_error')!=original['error']
          or recovered.get('id')!=original['id']
          or recovered.get('source_index')!=original['source_index']):
        raise ValueError('Recovery origin differs from original unavailable case')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();public=args.study/'public'
    plan=load(public/'plan.json');info=plan['recovery'];base=load(public/'integrity.json');summary=load(public/'summary.json')
    if base['status']!='pass':raise ValueError('Complete numerical integrity audit required first')
    bind(public/'plan.json',base['plan_sha256']);bind(public/'summary.json',base['summary_sha256'])
    raw=(public/'results.jsonl').read_bytes()
    if sha(raw)!=base['results_prefix_sha256'] or len(raw)!=base['results_prefix_bytes']:
        raise ValueError('Combined records differ from full audit')
    parent=Path(info['parent']);parent_plan,old_rows,missing=parent_state(parent)
    if {k:v for k,v in plan.items() if k!='recovery'}!=parent_plan:
        raise ValueError('Original sample, keys, score or protocol changed')
    if info['indices']!=missing or info['new_measurements_planned']!=len(missing) or info['retained_count']!=500-len(missing):
        raise ValueError('Recovery set differs from all unmeasured cases')
    if info['original_attempt_status']!='incomplete' or info['fresh_sample'] is not False:
        raise ValueError('Recovery cannot relabel the original attempt or sample')
    for name,digest in info['parent_files_sha256'].items():bind(parent/'public'/name,digest)
    bind(Path(__file__).with_name('recover_prompt_null.py'),info['runner_sha256'])
    bind(Path(__file__).with_name('preflight_prompt_null_memory.py'),info['preflight_runner_sha256'])
    pp=Path(info['preflight'])/'public'
    for name,digest in info['preflight_files_sha256'].items():bind(pp/name,digest)
    preplan=load(pp/'plan.json');preflight=load(pp/'summary.json')
    if preplan['parent']!=str(parent.resolve()) or preflight['status']!='pass' or preflight['new_source_tasks_measured']!=0:
        raise ValueError('Preflight belongs to another parent or did not pass')
    bind(parent/'public/integrity.json',preplan['parent_audit_sha256'])
    if (preplan['cache_limit_bytes']!=0 or info['cache_limit_bytes']!=0 or not info['clear_between_documents']
            or not preplan['clear_between_documents'] or load(pp/'cache-policy.json')['applied_limit_bytes']!=0):
        raise ValueError('Cache-disabled policy differs')
    ranks=[]
    for old in old_rows:
        if 'error' not in old:
            heads=load(parent/(old['id']+'.heads.json'))
            ranks.append((len(heads['conditioning_prefix_ids'])+len(heads['token_ids']),old['id']))
    ranks.sort();expected=[ranks[i][1] for i in (0,len(ranks)//2,len(ranks)-1)]
    if preplan['cases']!=expected or [r['id'] for r in preflight['cases']]!=expected:
        raise ValueError('Preflight size-based selection differs')
    preflight_heads=0
    for row in preflight['cases']:
        if not row['passed'] or not row['heads_exact'] or not row['scores_exact'] or row['cache_bytes']!=0:
            raise ValueError('Preflight parity or cache check failed')
        for suffix in ('.heads.json','.scores.json'):
            if load(pp.parent/(row['id']+suffix))!=load(parent/(row['id']+suffix)):
                raise ValueError('Preflight receipts differ from original values')
        preflight_heads+=row['heads']
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=len(old_rows):raise ValueError('All combined records required')
    recovered_available=0;new_bytes=0;memory=[]
    for original,row in zip(old_rows,rows,strict=True):
        check_record(original,row)
        if 'error' in original and 'error' not in row:
            recovered_available+=1
            if row['memory']['cache_bytes']!=0:raise ValueError('New record retained an allocator cache')
            memory.append(row['memory'])
            new_bytes+=sum((args.study/(row['id']+suffix)).stat().st_size for suffix in ('.heads.json','.scores.json'))
    expected_fields={'recovered_study':True,'retained_from_original':500-len(missing),
        'recovery_attempts':len(missing),'recovered_available':recovered_available,
        'original_attempt_status':'incomplete','original_screen_passed':False,
        'original_attempt_unchanged':True,'fresh_sample':False}
    if any(summary.get(k)!=v for k,v in expected_fields.items()):raise ValueError('Recovery summary overstates completion or freshness')
    if new_bytes>info['storage_budget_bytes']:raise ValueError('New receipts exceeded reserved storage budget')
    result={'status':'pass','scope':__doc__,'original_attempt_unchanged':True,'original_screen_passed':False,
        'fresh_sample':False,'retained_records':500-len(missing),'recovery_attempts':len(missing),
        'recovered_available':recovered_available,'preflight_exact_heads':preflight_heads,
        'new_head_and_score_bytes':new_bytes,'storage_budget_bytes':info['storage_budget_bytes'],
        'maximum_tracked_mlx_peak_bytes':max((m['peak_bytes'] for m in memory),default=None),
        'minimum_observed_disk_free_bytes':min((m['disk_free_bytes'] for m in memory),default=None),
        'plan_sha256':sha((public/'plan.json').read_bytes()),'summary_sha256':sha((public/'summary.json').read_bytes()),
        'numerical_audit_sha256':sha((public/'integrity.json').read_bytes()),
        'parent_audit_sha256':sha((parent/'public/integrity.json').read_bytes()),
        'audit_source_sha256':sha(Path(__file__).read_bytes()),
        'helper_sha256':sha(Path(__file__).with_name('recover_prompt_null.py').read_bytes()),
        'deployment_calibrated':False,'sdk_promotion':False}
    with (public/'recovery-integrity.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
