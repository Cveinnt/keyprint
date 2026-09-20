"""Operational completion of a frozen null sample after disk-only failures.

Retain every measured parent case without rerunning it. Measure every and only
previously unmeasured disk-guard case with unchanged source, keys and score.
The original attempt remains incomplete; this is not a fresh independent sample.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

from prompt_confirmation_selection import sha
from prompt_null_selection import prompt
from prompt_conditioned_likelihood import measure
from surrogate_likelihood import score
from validate_prompt_null import summarize, DISK_FLOOR
from develop_surrogate_likelihood import write

DISK_ERROR = {'type':'OSError','message':'Less than 2 GiB free; sample unavailable without retry'}


def load(path): return json.loads(path.read_text())


def bind(path, digest):
    if sha(path.read_bytes()) != digest: raise ValueError(f'Frozen file differs: {path.name}')


def parent_state(study):
    public=study/'public';plan=load(public/'plan.json');audit=load(public/'integrity.json');summary=load(public/'summary.json')
    if audit['status']!='pass' or summary['status']!='incomplete' or summary['fatal'] is not None:
        raise ValueError('Complete integrity audit of an incomplete, nonfatal parent required')
    bind(public/'plan.json',audit['plan_sha256']);bind(public/'summary.json',audit['summary_sha256'])
    raw=(public/'results.jsonl').read_bytes()
    if sha(raw)!=audit['results_prefix_sha256'] or len(raw)!=audit['results_prefix_bytes']:
        raise ValueError('Parent result log differs')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=500 or len(plan['tasks'])!=500:raise ValueError('Entire fixed 500-task parent required')
    missing=[]
    for index,(task,row) in enumerate(zip(plan['tasks'],rows,strict=True)):
        if row['id']!=f'null-{index:03d}' or any(row.get(k)!=v for k,v in task.items()):
            raise ValueError('Parent task identity differs')
        hp=study/(row['id']+'.heads.json');sp=study/(row['id']+'.scores.json')
        if 'error' in row:
            if row['error']!=DISK_ERROR or hp.exists() or sp.exists() or 'flagged' in row:
                raise ValueError('Only disk-guard failures before any measurement are recoverable')
            missing.append(index)
        else:
            bind(hp,row['heads_sha256']);bind(sp,row['scores_sha256'])
    if not missing or len(missing)!=summary['unavailable'] or len(rows)-len(missing)!=summary['available']:
        raise ValueError('Parent availability differs')
    return plan,rows,missing


def compact(path,value):
    with path.open('x') as stream:json.dump(value,stream,separators=(',',':'),allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','model','preflight','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();parent=args.study/'public'
    plan,originals,missing=parent_state(args.study)
    bind(args.source,plan['source_sha256'])
    bind(Path(__file__).with_name('validate_prompt_null.py'),plan['script_sha256'])
    for name,digest in plan['dependencies_sha256'].items():bind(Path(__file__).with_name(name),digest)
    pp=args.preflight/'public';preflight=load(pp/'summary.json');preplan=load(pp/'plan.json')
    if (preflight['status']!='pass' or preflight['new_source_tasks_measured']!=0
            or preplan['parent']!=str(args.study.resolve()) or preplan['cache_limit_bytes']!=0
            or not preplan['clear_between_documents'] or len(preflight['cases'])!=3
            or preflight['fatal'] is not None or not all(r['passed'] and r['heads_exact'] and r['scores_exact']
                and r['cache_bytes']==0 and r['disk_free_bytes']>=DISK_FLOOR for r in preflight['cases'])):
        raise ValueError('Exact cache-disabled preflight required')
    bind(parent/'integrity.json',preplan['parent_audit_sha256'])
    bind(Path(__file__).with_name('preflight_prompt_null_memory.py'),preplan['script_sha256'])
    source=[json.loads(line) for line in args.source.read_bytes().splitlines()]
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding=runtime_binding(max_steps=2048)
    total_tokens=sum(len(binding.encode_visible(source[originals[i]['source_index']]['response'])) for i in missing)
    # Compact JSON: conservative allowance for 100 sparse probabilities, two
    # term lists, prefix IDs and row metadata. No source or scores are discarded.
    storage_budget=total_tokens*6000+len(missing)*65536
    if shutil.disk_usage(args.output.parent).free < DISK_FLOOR+storage_budget:
        raise OSError('Insufficient disk headroom for bounded remaining receipts plus original guard')
    args.output.mkdir(mode=0o700);public=args.output/'public';public.mkdir()
    parent_files={n:sha((parent/n).read_bytes()) for n in ('plan.json','summary.json','integrity.json','identity.json','results.jsonl')}
    recovery={'scope':__doc__,'parent':str(args.study.resolve()),'parent_files_sha256':parent_files,
        'runner_sha256':sha(Path(__file__).read_bytes()),'preflight':str(args.preflight.resolve()),
        'preflight_files_sha256':{n:sha((pp/n).read_bytes()) for n in ('plan.json','summary.json','cache-policy.json')},
        'preflight_runner_sha256':preplan['script_sha256'],'cache_limit_bytes':0,'clear_between_documents':True,
        'indices':missing,'retained_count':len(originals)-len(missing),'new_measurements_planned':len(missing),
        'storage_format':'Compact JSON for new heads/scores; lossless parsed values',
        'storage_budget_bytes':storage_budget,'planned_literal_tokens':total_tokens,
        'original_attempt_status':'incomplete','fresh_sample':False,
        'failure_rule':'One recovery attempt per previously unmeasured case; no further retries or replacements. Parent stays unchanged.',
        'acceptance':'Combined all-500 observation screen only; original no-retry protocol remains failed'}
    write(public/'plan.json',{**plan,'recovery':recovery})
    shutil.copyfile(parent/'identity.json',public/'identity.json')
    for row in originals:
        if 'error' not in row:
            for suffix in ('.heads.json','.scores.json'):
                # Exclusive creation; retained files are never opened for write.
                os.link(args.study/(row['id']+suffix),args.output/(row['id']+suffix))
    print(json.dumps({'stage':'frozen','new_measurements':len(missing),'retained':len(originals)-len(missing),'storage_budget_bytes':storage_budget}),flush=True)
    started=time.monotonic();rows=[];setup_error=None;backend=None;recovery_count=0
    try:
        import mlx.core as mx
        from keyprint.backends.mlx import MLXModel,ASSETS
        from keyprint import sampling
        expected_identity={'model_assets':ASSETS,'profile':binding.profile.identity_receipt(),'sampling':sampling.identity()}
        if load(parent/'identity.json')!=expected_identity:raise ValueError('Model/profile/sampling identity differs')
        mx.set_cache_limit(0)
        backend=MLXModel.load(args.model)
        keys=[(Path(plan['confirmation'])/f'owner-{i}.key').read_bytes() for i in range(2)]
        if [sha(k) for k in keys]!=plan['key_commitments']:raise ValueError('Confirmation keys differ')
    except Exception as exc:setup_error={'type':type(exc).__name__,'message':str(exc)}
    with (public/'results.jsonl').open('x') as stream:
        for index,original in enumerate(originals):
            if 'error' not in original:
                row={**original,'execution_origin':'retained'}
            else:
                recovery_count+=1
                row={'id':original['id'],**plan['tasks'][index],'execution_origin':'recovery','original_error':original['error']}
                begin=time.monotonic()
                try:
                    if setup_error is not None:raise RuntimeError('Model setup failed; see summary fatal field')
                    if shutil.disk_usage(args.output).free < DISK_FLOOR:
                        raise OSError('Less than 2 GiB free during recovery; no further retry')
                    item=source[row['source_index']]
                    heads=measure(backend,binding,item['response'],prompt(item))
                    values=[score(binding.profile,key,heads) for key in keys]
                    hp=args.output/(row['id']+'.heads.json');sp=args.output/(row['id']+'.scores.json')
                    compact(hp,heads);compact(sp,values)
                    row.update(working_log_ratios=[v['working_log_ratio'] for v in values],
                        flags=[v['flagged'] for v in values],flagged=any(v['flagged'] for v in values),
                        scored_events=values[0]['scored_events'],outside_support=values[0]['outside_support'],
                        tokens=len(heads['token_ids']),heads_sha256=sha(hp.read_bytes()),scores_sha256=sha(sp.read_bytes()))
                except Exception as exc:row['error']={'type':type(exc).__name__,'message':str(exc)}
                if backend is not None:
                    gc.collect();mx.clear_cache()
                    row['memory']={'cache_bytes':mx.get_cache_memory(),'active_bytes':mx.get_active_memory(),
                        'peak_bytes':mx.get_peak_memory(),'disk_free_bytes':shutil.disk_usage(args.output).free}
                row['seconds']=time.monotonic()-begin
                print(json.dumps({'id':row['id'],'recovered':recovery_count,'flagged':row.get('flagged'),'error':row.get('error'),'memory':row.get('memory')}),flush=True)
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    for name,digest in parent_files.items():bind(parent/name,digest)
    result=summarize(rows,setup_error)
    result.update(seconds=time.monotonic()-started,recovered_study=True,
        retained_from_original=recovery['retained_count'],recovery_attempts=len(missing),
        recovered_available=sum(r['execution_origin']=='recovery' and 'error' not in r for r in rows),
        original_attempt_status='incomplete',original_screen_passed=False,original_attempt_unchanged=True,fresh_sample=False)
    write(public/'summary.json',result);print(json.dumps(result),flush=True)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
