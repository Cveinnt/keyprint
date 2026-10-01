"""One explicit continuation after a verified pre-generation second-key stop."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from validate_specialized import sha, save, identities, REVISION
from specialized_oracle import evaluate


def main(root, model):
    plan=json.loads((root/'public/plan.json').read_text())
    commit=json.loads((root/'public/commitment.json').read_text())
    rows=json.loads((root/'private/runs.json').read_text())
    stopped=json.loads((root/'supervisor-clean-assets/result.json').read_text())
    if (sha(root/'public/plan.json')!=commit['plan_sha256'] or identities()!=plan['source_sha256']
            or model.name!=REVISION or len(rows)!=16 or len(plan['schedule'])!=32
            or stopped.get('reason')!='system_pressure' or not stopped.get('cleanup_verified')):
        raise ValueError('Continuation preconditions changed')
    for i,row in enumerate(rows):
        if row['attempt']!=i or any(row[k]!=v for k,v in plan['schedule'][i].items()):
            raise ValueError('Completed prefix changed')
    for i in range(16,32):
        if (root/'private'/f'attempt-{i:02d}').exists():raise ValueError('Never retry an existing attempt')
    key=(root/'private/key-1').read_bytes()
    if hashlib.sha256(key).hexdigest()!=plan['key_sha256'][1]:raise ValueError('Key changed')
    record={'plan_sha256':commit['plan_sha256'],'prefix_runs_sha256':sha(root/'private/runs.json'),
        'original_stop_sha256':sha(root/'supervisor-clean-assets/result.json'),
        'continuation_script_sha256':sha(Path(__file__)),'first_attempt':16,'last_attempt':31,
        'reason':'Original worker stopped before any second-key attempt. Fresh process avoids retaining prior-key model state. All old outputs unchanged.'}
    with (root/'public/continuation.json').open('x') as stream:json.dump(record,stream,indent=2)
    (root/'private/prefix-runs.json').write_bytes((root/'private/runs.json').read_bytes())
    from keyprint import Keyprint
    cases={c['id']:c for c in plan['cases']}
    with Keyprint.from_mlx(model,key=key,temperature=.7,top_k=100) as wm:
        save(root/'public/identity-1.json',wm.identity)
        for i in range(16,32):
            row=dict(plan['schedule'][i],attempt=i);start=time.monotonic()
            if row['key_slot']!=1:raise ValueError('Unexpected continuation key')
            try:
                result=wm.generate(cases[row['case']]['prompt'],condition=row['condition'],
                    max_tokens=192,trace=True,output=root/'private'/f'attempt-{i:02d}')
                row.update(text=result.text,completion=result.report['payload']['completion'],
                           tokens=len(result.report['payload']['committed_token_ids']))
                row['exact_trace']=''.join(t.text for t in result.trace)==result.text
                row['oracle']=evaluate(result.text,cases[row['case']])
                row['pass']=row['exact_trace'] and row['completion']=='eos' and row['oracle']['status']=='pass'
                m=wm.inspect(result.text)
                row['diagnostic']={'ones':m.ones,'trials':m.trials,'events':m.events,'fraction':m.fraction,'calibrated':False}
            except Exception as exc:
                row.update(error_type=type(exc).__name__,error=str(exc)[:300],**{'pass':False})
            row['elapsed_seconds']=time.monotonic()-start;rows.append(row)
            save(root/'private/runs.json',rows)
            print(f"{len(rows)}/32 {row['case']} {row['condition']} pass={row['pass']} tokens={row.get('tokens')}",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);args=p.parse_args();main(args.output,args.model)
