"""Fresh paired inference for the frozen two-key weighted-reference rule.

Select twelve new source tasks by category metadata, not observed scores. Retain
all outputs, errors, truncations and short answers. Not a quality approval.
"""
import argparse
import json
import os
from pathlib import Path
import time

from keyprint import Keyprint
from layer_likelihood import extract
from validate_corpus_power import select_tasks
from validate_null_corpus import SOURCE_SHA256, sha, write
from weighted_null import reference_tail, two_key_tail, WEIGHTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--null-study',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.source.read_bytes()
    if sha(raw)!=SOURCE_SHA256: raise ValueError('Pinned corpus differs')
    null_plan_path=args.null_study/'public/plan.json'
    null_plan=json.loads(null_plan_path.read_text())
    if null_plan['target']!=.01 or null_plan['weights_integer']!=list(WEIGHTS):
        raise ValueError('Frozen reference rule differs')
    for name in ('weighted_null.py','layer_likelihood.py'):
        if sha(Path(__file__).with_name(name).read_bytes())!=null_plan['dependencies_sha256'][name]:
            raise ValueError('Frozen reference implementation differs')
    selection={'records':[{**r,'split':'heldout'} for r in null_plan['records']]}
    tasks=select_tasks([json.loads(line) for line in raw.splitlines()],selection)
    keys=[(args.null_study/f'owner-{i}.key').read_bytes() for i in range(2)]
    if [sha(k) for k in keys]!=null_plan['key_commitments']: raise ValueError('Null key commitments differ')
    args.output.mkdir(mode=0o700);public=args.output/'public';public.mkdir()
    for i,key in enumerate(keys):
        with os.fdopen(os.open(args.output/f'owner-{i}.key',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f: f.write(key)
    plan={'scope':'Fresh model-output weighted-reference screen; no quality or deployment acceptance',
          'script_sha256':sha(Path(__file__).read_bytes()),'null_plan_sha256':sha(null_plan_path.read_bytes()),
          'source_sha256':SOURCE_SHA256,'source':null_plan['source'],'revision':null_plan['revision'],
          'attribution':null_plan['attribution'],'license':null_plan['license'],'tasks':tasks,
          'key_commitments':null_plan['key_commitments'],'target':.01,'weights_integer':list(WEIGHTS),
          'temperature':.7,'top_k':100,'max_tokens':1024,
          'selection':'First two eligible source tasks in each of six fixed categories from the new 500 groups; original prompts unchanged',
          'ordering':'Ordinary first on even index; marked first on odd; key index is task index modulo two',
          'comparison':'Two-key family tail <= .01; matching and wrong-key hits use per-key <= .005; no tuning or retries',
          'failure_rule':'Retain and count all errors, incomplete/short/long texts; no silent filtering',
          'length_scope':'Null controls were 100-400 words; generated lengths can differ and remain in the report'}
    write(public/'plan.json',plan)
    rows=[];failure=None
    try:
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        binding=runtime_binding(max_steps=2048)
        candidates=[Keyprint.from_mlx(args.model,key=k) for k in keys]
        if candidates[0].identity!=candidates[1].identity: raise ValueError('Model profiles differ')
        write(public/'identity.json',candidates[0].identity)
        for i,task in enumerate(tasks):
            for condition in (('ordinary','marked') if i%2==0 else ('marked','ordinary')):
                name=f'weighted-{i:02d}-{condition}';started=time.monotonic()
                row={'id':name,'condition':condition,'key_index':i%2,'source_index':task['source_index'],'category':task['category']}
                try:
                    candidate=candidates[i%2]
                    generated=candidate.generate(task['prompt'],condition=condition,max_tokens=1024,output=args.output/name)
                    row.update(text=generated.text,completion=generated.report['payload']['completion'],
                               usage=generated.report['usage'],words=len(generated.text.split()))
                    values=[]
                    for key in keys:
                        bits=extract(binding,generated.text,key)
                        counts=candidate.inspect(generated.text,key=key)
                        if int(bits.sum())!=counts.ones or bits.size!=counts.trials: raise ValueError('SDK event-count replay differs')
                        result=reference_tail(bits);other=reference_tail(bits,double_grid=True)
                        if abs(result['reference_tail']-other['reference_tail'])>1e-10: raise ArithmeticError('Unstable FFT grid')
                        values.append(result)
                    family=two_key_tail([v['reference_tail'] for v in values])
                    row.update(weighted=values,family_reference_tail=family,any_key_flagged=family<=.01,
                               matching_flagged=values[i%2]['reference_tail']<=.005,
                               other_flagged=values[1-i%2]['reference_tail']<=.005)
                except Exception as exc: row['error']=type(exc).__name__
                row['seconds']=time.monotonic()-started;rows.append(row);write(public/(name+'.json'),row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('text','weighted')}),flush=True)
    except Exception as exc: failure=type(exc).__name__
    finally:
        errors=sum('error' in r for r in rows)
        summary={'status':'completed' if len(rows)==24 and not errors and not failure else 'incomplete',
                 'attempts':len(rows),'errors':errors,'failure':failure,
                 'groups':{c:{'planned':12,'available':sum(r['condition']==c and 'error' not in r for r in rows),
                              'matching':sum(r['condition']==c and r.get('matching_flagged',False) for r in rows),
                              'other':sum(r['condition']==c and r.get('other_flagged',False) for r in rows),
                              'any_key':sum(r['condition']==c and r.get('any_key_flagged',False) for r in rows)} for c in ('ordinary','marked')},
                 'truncated':sum(r.get('completion')=='length' for r in rows),
                 'outside_null_word_range':sum('words' in r and not 100<=r['words']<=400 for r in rows),
                 'deployment_calibrated':False,'quality_acceptance':False,'scope':plan['scope']}
        write(public/'summary.json',summary);print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='completed' else 1


if __name__=='__main__': raise SystemExit(main())
