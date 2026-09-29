"""Paired synthetic kernel/reference verification and timing, not serving evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from paced_integer_kernel import PacedKernel
from paced_reserved_reference import step as reference
from paced_budget_float_reference import normalized


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;out.mkdir(mode=0o700)
    save=lambda n,x:(out/n).write_text(json.dumps(x,indent=2)+'\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    root=Path(__file__).parent
    files=[root/n for n in ['benchmark_paced_kernel.py','paced_integer_kernel.py',
        'paced_reserved_reference.py','paced_budget_float_reference.py','predictable_budget_oracle.py']]
    files.append(root.parent/'src/keyprint/_engine/research/keyprint_exact_categorical_v2.py')
    hashes={str(f.relative_to(root.parent)):digest(f) for f in files}
    cases=[{'kind':kind,'size':n,'seed':seed} for n in (100,1000)
        for kind in ('uniform','zipf','wide-tail','grouped') for seed in (0,1,2,3)]
    plan={'schema':'keyprint.paced-kernel-benchmark.v1','cases':cases,'layers':30,
        'policy':'reserved-base-relative-margin-v1','rounding_margin':'2^-40',
        'bounds':'half-to-double total; one-eighth-base per layer; exact categorical checks',
        'relative_comparison_tolerance':'1e-10','source_sha256':hashes,
        'timing':'Pair separate fast/reference trajectories; alternate call order each layer; time calls only; setup and comparison excluded',
        'selection':'Same 32 fixed numerical cases; no retries or dropped failures',
        'scope':'Synthetic numerical correctness and local microbenchmark only, no model, detector or serving qualification'}
    save('plan.json',plan);results=[]
    for case in cases:
        n,kind=case['size'],case['kind']
        base=tuple(1. if kind=='uniform' else 1./(i+1) if kind=='zipf' else
            math.exp(-600*i/(n-1)) if kind=='wide-tail' else float(1+2*(i%2)) for i in range(n))
        kernel=PacedKernel(base);pbase=normalized(base);fast=slow=base
        rng=random.Random(case['seed']);fast_ns=[];slow_ns=[]
        result=dict(case,passed=False,steps=0,maximum_base_relative_difference=0.,
            fast_freezes=0,reference_freezes=0)
        try:
            for layer in range(30):
                labels=[rng.randrange(2) for _ in range(n//2 if kind=='grouped' else n)]
                bits=[labels[i//2] for i in range(n)] if kind=='grouped' else labels
                for which in (('fast','reference') if layer%2 else ('reference','fast')):
                    start=time.perf_counter_ns()
                    if which=='fast':
                        fast,receipt=kernel.step(fast,bits);fast_ns.append(time.perf_counter_ns()-start)
                        result['fast_freezes']+=receipt['near_boundary_freeze']
                    else:
                        slow,receipt=reference(base,slow,bits);slow_ns.append(time.perf_counter_ns()-start)
                        result['reference_freezes']+=receipt['near_boundary_freeze']
                difference=max(abs(a-b)/p_i for a,b,p_i in zip(normalized(fast),normalized(slow),pbase))
                result['maximum_base_relative_difference']=max(result['maximum_base_relative_difference'],float(difference))
                if difference*10**10>1: raise ArithmeticError('Kernel/reference trajectory difference exceeds fixed tolerance')
                result['steps']+=1
            result['passed']=True
        except (ValueError,ArithmeticError) as exc:
            result.update(error_type=type(exc).__name__,error=str(exc))
        result.update(fast_nanoseconds=fast_ns,reference_nanoseconds=slow_ns,
            median_fast_ms=statistics.median(fast_ns)/1e6 if fast_ns else None,
            median_reference_ms=statistics.median(slow_ns)/1e6 if slow_ns else None)
        results.append(result);save('progress.json',results)
        print(f"Paired case {len(results)}/32: {kind}/{n} seed {case['seed']} passed={result['passed']}",flush=True)
    if any(digest(f)!=hashes[str(f.relative_to(root.parent))] for f in files):
        raise ValueError('Benchmark source changed during execution')
    save('results.json',{'plan_sha256':digest(out/'plan.json'),'cases':results,
        'attempts':len(results),'passed':sum(r['passed'] for r in results),
        'verified_steps':sum(r['steps'] for r in results),'model_inference':False,
        'serving_qualified':False,'quality_acceptance':False,'detector_calibrated':False,'sdk_promotion':False})
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__': raise SystemExit(main())
