"""Retained synthetic session/draw replay, explicitly not language-model inference."""
import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from paced_source_session import PacedProfile,PacedSourceSession,Config,SourceRequest,policy_spec
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import replay_events


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;out.mkdir(mode=0o700)
    save=lambda n,x:(out/n).write_text(json.dumps(x,indent=2)+'\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    root=Path(__file__).parent
    names=['tools/validate_paced_session.py','tools/paced_source_session.py','tools/paced_integer_kernel.py',
        'src/keyprint/_engine/research/keyprint_exact_categorical_v2.py',
        'src/keyprint/_engine/legacy/_impl/research/grouped_canonical_prototype.py',
        'src/keyprint/_engine/legacy/_impl/research/token_source_policy.py',
        'src/keyprint/_engine/legacy/_impl/research/byte_trie_source_policy.py']
    hashes={n:digest(root.parent/n) for n in names}
    prefix=b'abcdefghijklmnopqrstuvwx'
    pieces=(prefix,b'A',b' A',b'B',b'C',b' \n',None)
    profile=PacedProfile(pieces,tokenizer_identity='synthetic-session-v1',eos_ids=[6],
                         config=Config(history=1,layers=30,max_steps=33))
    cases=[{'purpose':purpose,'key_slot':key,'condition':condition} for purpose in ('general','proofread')
           for key in range(4) for condition in ('ordinary','marked')]
    plan={'schema':'keyprint.paced-session-validation.v1','cases':cases,'max_steps':33,
        'policy':policy_spec(),'profile_sha256':profile.digest,'source_sha256':hashes,
        'selection':'All16 synthetic cases, four public test keys, both arms, general/proofread; retain EOS and cap endings',
        'fixture':'One deterministic prefix then fixed supplied probabilities; not a model or quality corpus',
        'key_fixture':'32 repetitions of key_slot byte; public test data, never production keys',
        'randomness':'Python Random seeded by case index, test-only, no production security claim'}
    save('plan.json',plan);results=[]
    for index,case in enumerate(cases):
        key=bytes([case['key_slot']])*32;rng=random.Random(index)
        request=SourceRequest() if case['purpose']=='general' else SourceRequest('proofread',(prefix+b'A'+prefix+b'B').decode())
        s=PacedSourceSession(profile,key,condition=case['condition'],request=request)
        trace=[];ids=[];events=[];result=dict(case,passed=False,ending=None)
        try:
            for step_index in range(33):
                q=np.array([1.,0.,0.,0.,0.,0.,0.]) if step_index==0 else np.array([0.,.15,.15,.2,.25,.15,.1])
                prepared=s.prepare(q);decision=s.last_decision
                exact=tuple(F(float(x)) for x in q);total=sum(exact)
                base={i:exact[i]/total for i in prepared.token_ids}
                actual={i:F(w,prepared.distribution.total) for i,w in zip(prepared.token_ids,prepared.distribution.weights)}
                excluded={i for i in actual if profile.classes[i] is None or i in decision['protected_token_ids']}
                assert sum(actual.values())==1 and all(base[i]/2<=actual[i]<=2*base[i] for i in actual)
                assert all(actual[i]==base[i] for i in excluded)
                if case['condition']=='ordinary' or not decision['partitioned']:assert actual==base
                draw=s.draw(prepared,rng.getrandbits)
                # Independent rational CDF check, not the integer lookup helper.
                point=F(draw.integer_point,draw.total_weight);cumulative=F(0);selected=None
                for i in prepared.token_ids:
                    cumulative+=actual[i]
                    if point<cumulative:selected=i;break
                assert selected==draw.token_index
                assert all(not t.accepted and t.value>=draw.total_weight for t in draw.transcript[:-1])
                if draw.transcript:assert draw.transcript[-1].accepted
                event=s.commit(prepared,draw.token_index);ids.append(draw.token_index)
                if event is not None:events.append(event)
                trace.append({'step':step_index,'base':q.tolist(),'token_ids':prepared.token_ids,
                    'integer_distribution':asdict(prepared.distribution),'decision':decision,'draw':asdict(draw)})
                if draw.token_index in profile.eos_ids:result['ending']='eos';break
            if result['ending'] is None:s.close();result['ending']='cap'
            assert tuple(events)==replay_events(profile,key,ids)
            result.update(passed=True,committed_steps=len(ids),scored_events=len(events),receipt=s.source_receipt())
        except Exception as error:
            result.update(error_type=type(error).__name__,error=str(error),committed_steps=len(ids))
        save(f'case-{index:02}.json',trace)
        result['trace_sha256']=digest(out/f'case-{index:02}.json')
        results.append(result);save('progress.json',results)
        print(f"Session {index+1}/16: passed={result['passed']} steps={len(ids)} ending={result['ending']}",flush=True)
    if any(digest(root.parent/n)!=h for n,h in hashes.items()):raise ValueError('Source changed during study')
    save('results.json',{'plan_sha256':digest(out/'plan.json'),'cases':results,'attempts':len(results),
        'passed':sum(r['passed'] for r in results),'committed_steps':sum(r['committed_steps'] for r in results),
        'model_inference':False,'quality_acceptance':False,'detector_calibrated':False,'sdk_promotion':False})
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
