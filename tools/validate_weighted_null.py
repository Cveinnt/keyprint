"""Fresh 500-document/two-key screen of a predeclared weighted reference tail.

Uses previously unused exact-deduplicated source groups and new keys. No model
generation, fitted cutoff, retries or deployment verdict. Only public/ may be
shared. Source responses and watermark keys are never included in public/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

from scipy.stats import binom
from layer_likelihood import extract
from validate_null_corpus import SOURCE_SHA256, select_rows, iid_binomial_upper, write
from weighted_null import reference_tail, two_key_tail, WEIGHTS, ALIAS_TARGET, NUMERIC_MARGIN


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fresh_rows(corpus, prior):
    selected, summary = select_rows(corpus, 1500)
    if {r['source_index'] for r in selected[:1000]} != {r['source_index'] for r in prior['records']}:
        raise ValueError("Previously used exact-group selection differs")
    return selected[1000:], summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if sha(raw) != SOURCE_SHA256: raise ValueError('Pinned dataset checksum differs')
    prior = json.loads((args.prior_study/'public/plan.json').read_text())
    selected, selection = fresh_rows([json.loads(line) for line in raw.splitlines()], prior)
    keys = [secrets.token_bytes(32) for _ in range(2)]
    commitments = [sha(k) for k in keys]
    if len(set(commitments)) != 2 or any(c in sum(prior['key_commitments'].values(), []) for c in commitments):
        raise ValueError('New independent keys required')
    args.output.mkdir(mode=0o700); public=args.output/'public';public.mkdir()
    for i,key in enumerate(keys):
        with os.fdopen(os.open(args.output/f'owner-{i}.key',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
            f.write(key)
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding=runtime_binding(max_steps=2048)
    plan={'scope':'Fresh public-corpus reference-null validation; not production or quality qualification',
          'source':prior['source'],'revision':prior['revision'],'source_sha256':SOURCE_SHA256,
          'attribution':prior['attribution'],'license':prior['license'],
          'script_sha256':sha(Path(__file__).read_bytes()),
          'dependencies_sha256':{n:sha(Path(__file__).with_name(n).read_bytes()) for n in
                                ['weighted_null.py','layer_likelihood.py','validate_null_corpus.py']},
          'prior_plan_sha256':sha((args.prior_study/'public/plan.json').read_bytes()),
          'binding':binding.profile.identity_receipt(),'selection':selection,
          'records':[{k:v for k,v in r.items() if k!='text'} for r in selected],
          'key_commitments':commitments,'primary':'linear_10_to_1',
          'secondary':'uniform exact-binomial, descriptive only; no choose-either decision',
          'weights_integer':list(WEIGHTS),'alias_target':ALIAS_TARGET,'numeric_margin':NUMERIC_MARGIN,
          'decision':'2 * min(two key reference tails) <= .01; no fitted threshold',
          'target':.01,'planned_documents':500,'unit':'One document across two keys, not 1000 independent observations',
          'grid_check':'Double FFT size for every positive statistic; abort acceptance if difference exceeds 1e-10',
          'uncertainty':'97.5% one-sided binomial bounds assume IID documents; near duplicates, authors and topics can violate this. Ideal-PRF null assumptions and numerical margin are not fixed-key deployment guarantees.',
          'failure_rule':'Retain all attempts; no retries; missing tails are unavailable, not negatives'}
    write(public/'plan.json',plan)
    rows=[];started=time.monotonic()
    with (public/'results.jsonl').open('x') as stream:
        for item in selected:
            row={k:v for k,v in item.items() if k!='text'}
            try:
                weighted=[];uniform=[]
                for key in keys:
                    bits=extract(binding,item['text'],key)
                    result=reference_tail(bits)
                    other=reference_tail(bits,double_grid=True)
                    delta=abs(result['reference_tail']-other['reference_tail'])
                    if delta>1e-10: raise ArithmeticError('Reference tail is not grid stable')
                    result['double_grid_difference']=delta;weighted.append(result)
                    uniform.append(float(binom.sf(int(bits.sum())-1,bits.size,.5)))
                family={'linear_10_to_1':two_key_tail([s['reference_tail'] for s in weighted]),
                        'uniform':two_key_tail(uniform)}
                row.update(weighted=weighted,uniform=uniform,family_reference_tail=family,
                           flagged={k:v<=.01 for k,v in family.items()})
            except Exception as exc: row['error']=type(exc).__name__
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if len(rows)%50==0: print(json.dumps({'documents':len(rows),'errors':sum('error' in r for r in rows)}),flush=True)
    errors=sum('error' in r for r in rows)
    panels={}
    for name in ('linear_10_to_1','uniform'):
        available=[r for r in rows if 'error' not in r];hits=sum(r['flagged'][name] for r in available)
        panels[name]={'available':len(available),'planned':500,'unavailable':500-len(available),'false_hits':hits,
                      'iid_only_upper_97_5_percent':iid_binomial_upper(hits,500) if not errors else None}
    summary={'status':'completed' if not errors else 'incomplete','errors':errors,'panels':panels,
             'seconds':time.monotonic()-started,'deployment_calibrated':False,'power_established':False,
             'scope':plan['scope'],'uncertainty':plan['uncertainty']}
    write(public/'summary.json',summary);print(json.dumps(summary),flush=True)
    return 1 if errors else 0


if __name__=='__main__': raise SystemExit(main())
