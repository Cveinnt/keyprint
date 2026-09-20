"""Validate a new cap-finalization runtime against the retained real-model failure.

All original model heads and random draws must match. The original failed study
remains failed. This replay is not a fresh serving or quality confirmation.
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
    parser.add_argument('--expected-runtime', required=True)
    parser.add_argument('--postmortem', type=Path, required=True)
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
    postmortem=json.loads(args.postmortem.read_text())
    if postmortem['original_journal_sha256']!=sha(source/'journal.jsonl'):
        raise ValueError('Postmortem belongs to another original attempt')
    declaration={'original_journal_sha256':sha(source/'journal.jsonl'),
        'original_postmortem_sha256':sha(args.postmortem),
        'expected_runtime':args.expected_runtime, 'validator_sha256':sha(Path(__file__)),
        'scope':__doc__}
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2))
    import mlx.core as mx
    import numpy as np
    from keyprint import Keyprint
    from keyprint._engine.research.keyprint_candidate_v3_caller import DurableJournal
    key=(args.run/'owner.key').read_bytes()
    sdk=Keyprint.from_mlx(args.model,key=key,execution='experimental-fast')
    if (sdk.identity['runtime_profile_sha256']!=args.expected_runtime
            or args.expected_runtime==starts[0]['runtime_profile_sha256']):
        raise ValueError('Must use the separately declared new runtime')
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
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if len(captured)==1:
        (args.output/'receipt.json').write_text(json.dumps(captured[0].receipt(),indent=2,ensure_ascii=False))
    if (counts!={'heads':len(heads),'draws':len(draws)} or report['kind']!='generation_trace'
            or replay_events[-1]['outcome']!='length' or len(captured)!=1):
        raise ValueError('Original sampling did not finish at the declared cap')
    raw=captured[0]._raw
    carrier=raw._current
    pending=carrier.decoder.getstate()[0]
    final=captured[0].receipt()['final']
    if (carrier.text!=postmortem['decoded_prefix'] or pending.hex()!=postmortem['pending_utf8_hex']
            or len(raw.committed_token_ids)!=postmortem['committed_tokens']
            or list(raw.committed_token_ids[-6:])!=postmortem['last_token_ids']
            or final['visible']['text_score'] is not None
            or final['visible']['pending_utf8_hex']!=pending.hex()
            or final['visible']['events_match'] is not False
            or report['carrier_rendering'][0]['pending_utf8_hex']!=pending.hex()
            or report['payload']['completion']!='length'
            or report['literal_replay_status'][0]['availability']!='unavailable'):
        raise ValueError('Capped result lost bytes or misreported replay availability')
    sampled=b''.join(raw.binding.token_bytes[i] for i in raw.committed_token_ids)
    if carrier.text.encode()+pending!=sampled:
        raise ValueError('Sampled bytes differ from text plus retained suffix')
    for field in ('sampled_tokens','bit_requests','bit_values_obtained','bit_values_journaled','model_calls'):
        if replay_events[-1][field]!=terminal[field]:
            raise ValueError('Consumed-work count changed: '+field)
    result={'status':'pass','original_journal_sha256':sha(source/'journal.jsonl'),
            'original_runtime_sha256':starts[0]['runtime_profile_sha256'],
            'runtime_sha256':sdk.identity['runtime_profile_sha256'],'matched_model_heads':counts['heads'],
            'reused_random_draws':counts['draws'],'committed_tokens':len(raw.committed_token_ids),
            'pending_utf8_hex':pending.hex(),'decoded_prefix_sha256':hashlib.sha256(carrier.text.encode()).hexdigest(),
            'original_attempt_unchanged':sha(source/'journal.jsonl')==declaration['original_journal_sha256'],
            'scope':__doc__}
    (args.output/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
