"""Replay one retained caller failure with original draws and model-head hashes.

Diagnostic only. Never changes the original attempt or converts it to success.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_mlx_capacity import read_journal
from benchmark_serving import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if Path(args.attempt).name!=args.attempt:raise ValueError('Attempt must be a single directory name')
    args.output.mkdir(mode=0o700)
    plan=json.loads((args.run/'public/plan.json').read_text())
    row=json.loads((args.run/'public'/(args.attempt+'.json')).read_text())
    cases=[c for c in plan['cases'] if c['id']==row['case']]
    if len(cases)!=1 or 'error' not in row:raise ValueError('Exactly one original failed case required')
    source=args.run/args.attempt
    events=read_journal(source/'journal.jsonl')
    starts=[e for e in events if e['kind']=='response_started']
    heads=[e['raw_logits_sha256'] for e in events if e['kind']=='prepared_step']
    draws=[e for e in events if e['kind']=='random_bits_returned']
    terminal=events[-1]
    if len(starts)!=1 or terminal['error_type']!='UnicodeDecodeError' or terminal['phase']!='finish':
        raise ValueError('Expected retained UTF-8 finalization failure')
    import mlx.core as mx
    import numpy as np
    from keyprint import Keyprint
    from keyprint._engine.research.keyprint_candidate_v3_caller import DurableJournal
    key=(args.run/'owner.key').read_bytes()
    sdk=Keyprint.from_mlx(args.model,key=key,execution='experimental-fast')
    if sdk.identity['runtime_profile_sha256']!=starts[0]['runtime_profile_sha256']:
        raise ValueError('Replay must use original frozen runtime')
    ids=sdk._backend.encode_prompt(cases[0]['prompt'])
    encoded=json.dumps(ids,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(encoded).hexdigest()!=starts[0]['prompt_sha256']:raise ValueError('Original prompt differs')
    counts={'heads':0,'draws':0}
    class ObservedModel:
        def __getattr__(self,name):return getattr(sdk._backend.model,name)
        def __call__(self,ids,**kwargs):
            result=sdk._backend.model(ids,**kwargs)
            if ids.shape[-1]==1:
                raw=result[:,-1,:].astype(mx.float32);mx.eval(raw)
                actual=hashlib.sha256(np.array(raw).tobytes()).hexdigest()
                if counts['heads']>=len(heads) or actual!=heads[counts['heads']]:raise ValueError('Model head differs')
                counts['heads']+=1
            return result
    def bits(n):
        if counts['draws']>=len(draws) or n!=draws[counts['draws']]['bits']:raise ValueError('Random request differs')
        event=draws[counts['draws']];counts['draws']+=1
        return int(event['value_decimal'])
    captured=[]
    original=sdk._candidate._core.pipeline
    def observe(*a,**k):
        value=original(*a,**k);captured.append(value);return value
    sdk._candidate._core.pipeline=observe
    with DurableJournal(args.output/'replay-journal.jsonl') as journal:
        report=sdk._candidate.run_response(ObservedModel(),ids,key=key,condition=row['condition'],
            random_bits=bits,journal=journal,reserve=lambda *a:None,max_tokens=row['max_tokens'],
            max_model_calls=row['max_tokens']+8,allow_thinking=False,allow_tools=False)
    replay_events=read_journal(args.output/'replay-journal.jsonl')
    if (counts!={'heads':len(heads),'draws':len(draws)} or report['kind']!='error'
            or replay_events[-1]['error_type']!='UnicodeDecodeError' or len(captured)!=1):
        raise ValueError('Failure did not reproduce exactly')
    raw=captured[0]._raw
    carrier=raw._current
    pending=carrier.decoder.getstate()[0]
    result={'status':'reproduced','original_journal_sha256':sha(source/'journal.jsonl'),
            'runtime_sha256':sdk.identity['runtime_profile_sha256'],'matched_model_heads':counts['heads'],
            'reused_random_draws':counts['draws'],'committed_tokens':len(raw.committed_token_ids),
            'pending_utf8_hex':pending.hex(),'decoded_prefix':carrier.text,
            'last_token_ids':list(raw.committed_token_ids[-6:]),
            'last_token_bytes_hex':[raw.binding.token_bytes[i].hex() for i in raw.committed_token_ids[-6:]],
            'original_attempt_unchanged':True,'scope':__doc__}
    (args.output/'postmortem.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('decoded_prefix','scope')}))


if __name__=='__main__':main()
