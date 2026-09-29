"""Fixed low-memory numerical stress study; no model or statistical power claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import time

from paced_budget_float_reference import normalized, step


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output; out.mkdir(mode=0o700)
    save=lambda name,x:(out/name).write_text(json.dumps(x,indent=2)+'\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    names=['validate_paced_float.py','paced_budget_float_reference.py',
           'paced_budget_oracle.py','predictable_budget_oracle.py']
    root=Path(__file__).parent
    cases=[{'kind':kind,'size':n,'seed':seed} for n in (100,1000)
           for kind in ('uniform','zipf','wide-tail','grouped') for seed in (0,1,2,3)]
    plan={'schema':'keyprint.paced-float-stress.v1','layers':30,'cases':cases,
          'selection':'Fixed 32 synthetic cases, all retained; no model data, tuning or key-power inference',
          'source_sha256':{n:digest(root/n) for n in names},
          'binary64_output':True,'internal_arithmetic':'Exact Fraction reference, not fast float kernel',
          'scope':'Normalized represented probability bounds, support, rounding and numerical runtime only'}
    save('plan.json',plan); results=[]
    for case in cases:
        n,kind=case['size'],case['kind']
        base=tuple(1. if kind=='uniform' else 1./(i+1) if kind=='zipf' else
                   math.exp(-600*i/(n-1)) if kind=='wide-tail' else float(1+2*(i%2)) for i in range(n))
        original=normalized(base); q=base; rng=random.Random(case['seed'])
        result=dict(case,passed=False,steps=0,near_boundary_freezes=0,
                    minimum_ratio=1.,maximum_ratio=1.,maximum_rounding_tv=0.,maximum_group_ratio_error=0.)
        started=time.monotonic(); bit_hash=hashlib.sha256(); output_hash=hashlib.sha256()
        try:
            for _ in range(30):
                labels=[rng.randrange(2) for _ in range(n//2 if kind=='grouped' else n)]
                bits=[labels[i//2] for i in range(n)] if kind=='grouped' else labels
                bit_hash.update(bytes(bits))
                q,receipt=step(base,q,bits)
                output_hash.update(struct.pack('<'+str(n)+'d',*q))
                actual=normalized(q)
                ratios=[float(a/b) for a,b in zip(actual,original)]
                result['minimum_ratio']=min(result['minimum_ratio'],min(ratios))
                result['maximum_ratio']=max(result['maximum_ratio'],max(ratios))
                result['maximum_rounding_tv']=max(result['maximum_rounding_tv'],float(receipt['rounding_total_variation']))
                result['near_boundary_freezes']+=receipt['near_boundary_freeze']
                if kind=='grouped':
                    result['maximum_group_ratio_error']=max(result['maximum_group_ratio_error'],
                        max(abs(q[i+1]/q[i]/3-1) for i in range(0,n,2)))
                result['steps']+=1
            result['passed']=True
        except (ValueError,ArithmeticError) as exc:
            result.update(error_type=type(exc).__name__,error=str(exc))
        result.update(elapsed_seconds=time.monotonic()-started,
                      bits_sha256=bit_hash.hexdigest(),outputs_sha256=output_hash.hexdigest())
        results.append(result);save('progress.json',results)
        print(f"Numerical case {len(results)}/{len(cases)}: {kind}/{n} seed {case['seed']} passed={result['passed']}",flush=True)
    if any(digest(root/n)!=h for n,h in plan['source_sha256'].items()):
        raise ValueError('Reference source changed during execution')
    save('results.json',{'plan_sha256':digest(out/'plan.json'),'cases':results,
        'attempts':len(results),'passed':sum(r['passed'] for r in results),
        'certified_steps':sum(r['steps'] for r in results),
        'model_inference':False,'quality_acceptance':False,'detector_calibrated':False,'sdk_promotion':False})
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__': raise SystemExit(main())
