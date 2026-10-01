"""Frozen small code/structured-text pilot; sequential, original outputs retained."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets
import time

from specialized_oracle import evaluate

REVISION='545dc4251c05440727734bcd94334791f6ab0192'
HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def identities():
    return {str(p.relative_to(HERE.parent)):sha(p) for p in
        [HERE/'specialized_cases.json',HERE/'specialized_oracle.py',Path(__file__),
         *sorted((HERE.parent/'src/keyprint').rglob('*.py'))]}


def prepare(root,model):
    if model.name!=REVISION or not model.is_dir():raise ValueError('Require existing pinned Qwen3-8B snapshot')
    cases=json.loads((HERE/'specialized_cases.json').read_text())
    for case in cases:
        if evaluate(case['reference'],case)['status']!='pass':raise ValueError('Reference oracle failed')
    root.mkdir(mode=0o700);(root/'public').mkdir();(root/'private').mkdir(mode=0o700)
    keys=[secrets.token_bytes(32) for _ in range(2)]
    for i,key in enumerate(keys):
        path=root/'private'/f'key-{i}';path.write_bytes(key);path.chmod(0o600)
    plan={'schema':'keyprint.specialized-pilot.v1','model_revision':REVISION,
          'cases':cases,'schedule':[{'key_slot':key,'case':c['id'],'condition':arm}
            for key in range(2) for i,c in enumerate(cases)
            for arm in (('ordinary','marked') if (key+i)%2==0 else ('marked','ordinary'))],
          'max_tokens':192,'temperature':.7,'top_k':100,'source_sha256':identities(),
          'key_sha256':[hashlib.sha256(k).hexdigest() for k in keys],
          'dataset':'New author-created synthetic fixtures; MIT. No external benchmark or training-held-out claim.',
          'policy':'All 32 attempts, two fresh fixed keys, no retries, no stripping Markdown, no rewriting. EOS plus oracle pass required. Unsupported Python syntax is reported separately and counts as not passed.',
          'scope':'Development smoke test: two tasks each in Python, SQL, JSON and CSV. No powered noninferiority, blinded reader, calibrated detection, or serving-overhead acceptance.'}
    save(root/'public/plan.json',plan)
    save(root/'public/commitment.json',{'plan_sha256':sha(root/'public/plan.json')})
    print('Frozen 8 cases / 2 keys / 32 attempts',flush=True)


def run(root,model):
    plan=json.loads((root/'public/plan.json').read_text())
    commit=json.loads((root/'public/commitment.json').read_text())
    if sha(root/'public/plan.json')!=commit['plan_sha256'] or identities()!=plan['source_sha256']:
        raise ValueError('Frozen inputs or SDK changed')
    if (root/'private/runs.json').exists():raise ValueError('No reruns or overwriting attempts')
    if model.name!=REVISION:raise ValueError('Wrong model')
    from keyprint import Keyprint
    from keyprint.integrity import verify
    save(root/'public/sdk-integrity.json',verify())
    cases={c['id']:c for c in plan['cases']};rows=[]
    for slot in range(2):
        key=(root/'private'/f'key-{slot}').read_bytes()
        if hashlib.sha256(key).hexdigest()!=plan['key_sha256'][slot]:raise ValueError('Key changed')
        with Keyprint.from_mlx(model,key=key,temperature=.7,top_k=100) as wm:
            save(root/'public'/f'identity-{slot}.json',wm.identity)
            for i,attempt in enumerate(plan['schedule']):
                if attempt['key_slot']!=slot:continue
                row=dict(attempt,attempt=i);begin=time.monotonic()
                try:
                    result=wm.generate(cases[row['case']]['prompt'],condition=row['condition'],
                        max_tokens=plan['max_tokens'],trace=True,output=root/'private'/f'attempt-{i:02d}')
                    row.update(text=result.text,completion=result.report['payload']['completion'],
                               tokens=len(result.report['payload']['committed_token_ids']))
                    row['exact_trace']=''.join(t.text for t in result.trace)==result.text
                    row['oracle']=evaluate(result.text,cases[row['case']])
                    row['pass']=row['exact_trace'] and row['completion']=='eos' and row['oracle']['status']=='pass'
                    measurement=wm.inspect(result.text)
                    row['diagnostic']={'ones':measurement.ones,'trials':measurement.trials,
                                       'events':measurement.events,'fraction':measurement.fraction,
                                       'calibrated':False}
                except Exception as exc:
                    row.update(error_type=type(exc).__name__,error=str(exc)[:300],**{'pass':False})
                row['elapsed_seconds']=time.monotonic()-begin;rows.append(row)
                save(root/'private/runs.json',rows)
                print(f"{len(rows)}/32 {row['case']} {row['condition']} pass={row['pass']} tokens={row.get('tokens')} error={row.get('error_type')}",flush=True)
    if len(rows)!=len(plan['schedule']):raise ValueError('Incomplete study')
    for row,expected in zip(rows,plan['schedule']):
        if any(row[k]!=v for k,v in expected.items()):raise ValueError('Attempt order changed')
    groups={}
    for domain in ['all','python','sql','json','csv']:
        groups[domain]={arm:{'attempts':len(selected:=[r for r in rows if r['condition']==arm and
                     (domain=='all' or cases[r['case']]['domain']==domain)]),
                     'passes':sum(r['pass'] for r in selected),
                     'eos':sum(r.get('completion')=='eos' for r in selected),
                     'errors':sum('error_type' in r for r in selected)} for arm in ['ordinary','marked']}
    save(root/'public/results.json',{'plan_sha256':commit['plan_sha256'],'groups':groups,'runs':rows,
         'quality_acceptance':False,'detector_calibrated':False,'scope':plan['scope']})
    print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    (prepare if a.mode=='prepare' else run)(a.output,a.model)
