"""Fixed synthetic allocation comparison; no model or acceptance transfer."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics

from audit_paced_study import digest
from complement_paced import ComplementPacedKernel, reference_step, POLICY
from paced_integer_kernel import PacedKernel
from paced_budget_float_reference import normalized
from paced_information import metrics
from paced_source_session import integer_distribution
from paced_key_rank import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output
    if str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID'):
        raise RuntimeError('External watchdog required')
    out.mkdir(mode=0o700)
    cases=[{'kind':kind,'size':n,'seed':seed} for n in (100,1000)
           for kind in ('uniform','zipf','wide-tail','grouped','peaked') for seed in range(4)]
    names=('compare_complement_paced.py','complement_paced.py','paced_integer_kernel.py',
           'paced_budget_float_reference.py','predictable_budget_oracle.py','paced_information.py',
           'paced_source_session.py','paced_key_rank.py','audit_paced_study.py')
    hashes=lambda:{n:digest(Path(__file__).with_name(n)) for n in names}
    plan={'schema':'keyprint.complement-paced-comparison.v1','policy':POLICY,'layers':30,
        'cases':cases,'source_sha256':hashes(),'total_probability_bounds':['1/2','2'],
        'per_layer_base_fraction':'1/8','rounding_margin':'2^-40','relative_tolerance':'1e-10',
        'selection':'All 32 prior numerical fixtures plus eight predeclared 99%-peaked fixtures; no retries or favorable subset',
        'coupling':'Same ideal label arrays for both policies; grouped fixtures share labels within pairs',
        'measurements':'Final conditional KL and expected uniform bit agreement, not sampled detection or semantic quality',
        'scope':'Synthetic numerical development, not inference, calibration, serving or marginal proof for finite PRF keys',
        'model_inference':False,'sdk_promotion':False}
    write(out/'plan.json',plan);rows=[]
    with (out/'progress.jsonl').open('x') as progress:
        for case in cases:
            n,kind=case['size'],case['kind']
            base=tuple(1. if kind=='uniform' else 1/(i+1) if kind=='zipf' else
                math.exp(-600*i/(n-1)) if kind=='wide-tail' else float(1+2*(i%2)) if kind=='grouped'
                else .99 if i==0 else .01/(n-1) for i in range(n))
            new=ComplementPacedKernel(base);old=PacedKernel(base);current=reference=previous=base
            rng=random.Random(case['seed']);ones=[0]*n
            row=dict(case,passed=False,verified_steps=0,maximum_base_relative_difference=0.,
                     complement_freezes=0,previous_freezes=0)
            try:
                for layer in range(30):
                    labels=[rng.randrange(2) for _ in range(n//2 if kind=='grouped' else n)]
                    bits=[labels[i//2] for i in range(n)] if kind=='grouped' else labels
                    current,receipt=new.step(current,bits)
                    reference,_=reference_step(base,reference,bits)
                    previous,old_receipt=old.step(previous,bits)
                    error=max(abs(a-b)/p for a,b,p in zip(normalized(current),normalized(reference),normalized(base)))
                    if error*10**10>1:raise ArithmeticError('Independent reference trajectory differs')
                    row['maximum_base_relative_difference']=max(row['maximum_base_relative_difference'],float(error))
                    row['complement_freezes']+=receipt['near_boundary_freeze']
                    row['previous_freezes']+=old_receipt['near_boundary_freeze']
                    ones=[a+b for a,b in zip(ones,bits)];row['verified_steps']+=1
                ordinary=integer_distribution(base)
                for name,weights in [('complement',current),('previous',previous)]:
                    dist=integer_distribution(weights);m=metrics(ordinary,dist,0)
                    row[name]={'kl_to_base_nats':m['kl_marked_to_base'],
                        'total_variation':m['total_variation'],
                        'expected_matching_bit_fraction':sum(w*g for w,g in zip(dist.weights,ones))/(30*dist.total)}
                row['base_expected_bit_fraction']=sum(w*g for w,g in zip(ordinary.weights,ones))/(30*ordinary.total)
                row['passed']=True
            except (ArithmeticError,ValueError) as error:
                row.update(error_type=type(error).__name__,error=str(error))
            rows.append(row);progress.write(json.dumps(row)+'\n');progress.flush()
            print(json.dumps({'case':len(rows),'planned':len(cases),'passed':row['passed']}),flush=True)
    if hashes()!=plan['source_sha256']:raise ValueError('Source changed during comparison')
    # Never summarize a successful-only subset. Failed rows remain in results.
    grouped={}
    if all(r['passed'] for r in rows):
        for kind in ('uniform','zipf','wide-tail','grouped','peaked'):
            selected=[r for r in rows if r['kind']==kind]
            grouped[kind]={name:{metric:statistics.mean(r[name][metric] for r in selected)
                for metric in selected[0][name]} for name in ('previous','complement')}
    result={'schema':'keyprint.complement-paced-results.v1','plan_sha256':digest(out/'plan.json'),
        'rows':rows,'attempts':len(rows),'passed':sum(r['passed'] for r in rows),
        'verified_steps':sum(r['verified_steps'] for r in rows),'means_by_kind':grouped,
        'scope':plan['scope'],'model_inference':False,'detector_calibrated':False,
        'quality_acceptance':False,'sdk_promotion':False,'launch_ready':False}
    write(out/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return 0 if result['passed']==40 else 1


if __name__=='__main__':raise SystemExit(main())
