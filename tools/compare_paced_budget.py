"""Fixed exact comparison, written before evaluation; no model or SDK changes."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from predictable_budget_capacity import analyze, serializable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output
    out.mkdir(mode=0o700)
    write=lambda name,value: (out/name).write_text(json.dumps(value,indent=2)+'\n')
    digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    tools=Path(__file__).parent
    names=['compare_paced_budget.py','paced_budget_oracle.py',
           'predictable_budget_oracle.py','predictable_budget_capacity.py']
    plan={'schema':'keyprint.paced-budget-comparison.v1','layers':30,'ratio':2,
          'per_layer_base_fraction':'1/8','policies':['maximum','paced-eighth'],
          'fixtures':[['1/2','1/2'],['3/4','1/4'],['99/100','1/100'],['999/1000','1/1000']],
          'selection':'Same four pre-existing two-token fixtures; one predeclared paced policy, no sweep or best-result selection',
          'method':'All independent fair labels propagated by exact state mass; no dropped or sampled states',
          'source_sha256':{n:digest(tools/n) for n in names},
          'model_inference':False,'detector_calibrated':False,'quality_acceptance':False}
    write('plan.json',plan)
    rows=[]
    for fixture in plan['fixtures']:
        base=tuple(F(x) for x in fixture)
        rows.append({'base':fixture,'policies':{policy:serializable(analyze(base,policy=policy))
            for policy in plan['policies']}})
    if any(digest(tools/n)!=value for n,value in plan['source_sha256'].items()):
        raise ValueError('Analysis source changed during execution')
    write('results.json',{'plan_sha256':digest(out/'plan.json'),'rows':rows,
        'scope':'Exact small-support independent-label comparison, not inference, calibrated detection or semantic quality',
        'launch_ready':False})


if __name__=='__main__': main()
