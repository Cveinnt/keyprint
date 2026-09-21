"""Actual clients against the pinned native vLLM HTTP server, including disconnect.

Requires the recorded local CPU server; no mocked generation or hosted calls.
Outputs and native token selections are retained for independent reconciliation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from openai import OpenAI, BadRequestError
from anthropic import Anthropic


def main():
    if not __debug__: raise RuntimeError('validation needs assertions; do not use -O')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--attempt', default='attempt-1')
    parser.add_argument('--url',default='http://127.0.0.1:8832')
    args=parser.parse_args()
    root=args.root
    if not args.attempt.replace('-', '').isalnum(): raise ValueError('invalid attempt name')
    public=root/'public'/args.attempt; public.mkdir(parents=True)
    token=(root/'api.key').read_bytes().hex()
    report={'status':'running','outputs':[],'checks':{},'quality_acceptance':False,
            'scope':'Pinned CPU HTTP pilot; not idempotent, production or detector qualification',
            'versions':{n:importlib.metadata.version(n) for n in ['openai','anthropic']}}
    def save(): (public/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    def traces():return set((root/'traces').glob('*.jsonl'))
    def selections(path):
        events=[]; previous='0'*64
        for i,line in enumerate(path.read_bytes().splitlines(keepends=True)):
            row=json.loads(line)
            assert row['sequence']==i and row['previous_sha256']==previous
            previous=hashlib.sha256(line).hexdigest();events.append(row['event'])
        assert not any(e['phase']=='host_prefix_mismatch' for e in events)
        return [e['token_id'] for e in events if e['phase']=='selected_tentative'],events
    client=OpenAI(base_url=args.url+'/v1',api_key=token,max_retries=0,timeout=180)
    anthropic=Anthropic(base_url=args.url,api_key=token,default_headers={'Authorization':'Bearer '+token},max_retries=0,timeout=180)
    def generate(item):
        prompt,condition=item
        result=client.chat.completions.create(model='keyprint',messages=[{'role':'user','content':prompt}],
            max_tokens=64,temperature=1,top_p=1,
            extra_body={'top_k':-1,'return_token_ids':True,'vllm_xargs':{'keyprint_condition':condition}})
        return {'prompt':prompt,'condition':condition,'protocol':'openai',
                'response':result.model_dump()}
    try:
        assert client.models.list().data[0].id=='keyprint'
        before=traces()
        inputs=[(p,c) for p in ['Explain why a seed needs water in one sentence.',
                 'En français, demande à Maya de conserver la sauvegarde.'] for c in ['ordinary','marked']]
        with ThreadPoolExecutor(max_workers=4) as pool:
            report['outputs']=list(pool.map(generate,inputs))
        new=traces()-before
        assert len(new)==4
        unmatched={str(p.name):selections(p)[0] for p in new}
        for row in report['outputs']:
            ids=row['response']['choices'][0]['token_ids']
            assert ids
            matches=[name for name,selected in unmatched.items() if selected==ids]
            assert len(matches)==1,(ids,unmatched)
            row['journal']=matches[0];del unmatched[matches[0]]
        report['checks']['four_concurrent_requests_exact_native_token_paths']=True
        save()
        for prompt in ['Explain why a seed needs water in one sentence.',
                       'En español, pide a Maya que conserve la copia de seguridad.']:
            before=traces()
            result=anthropic.messages.create(model='keyprint',messages=[{'role':'user','content':prompt}],
                max_tokens=64,extra_body={'temperature':1.,'top_p':1.,'top_k':-1})
            new=traces()-before;assert len(new)==1
            selected,events=selections(next(iter(new)))
            assert result.content and result.usage.output_tokens==len(selected)
            assert events[0]['condition']=='marked'
            report['outputs'].append({'prompt':prompt,'condition':'marked','protocol':'anthropic',
                'response':result.model_dump(),'journal':next(iter(new)).name,
                'scope':'Native journal and usage checked; Anthropic response has no final token IDs'})
            save()
        report['checks']['anthropic_text_and_token_count']=True
        before=traces()
        try:
            client.chat.completions.create(model='keyprint',messages=[{'role':'user','content':'Hello'}],
                max_tokens=16,temperature=.5)
        except BadRequestError: pass
        else: raise AssertionError('unsupported sampling accepted')
        assert traces()==before
        report['checks']['unsupported_sampling_rejected_before_journal']=True
        before=traces();pieces=[];ids=[]
        stream=client.chat.completions.create(model='keyprint',messages=[{'role':'user','content':'Write a long story about a forest.'}],
            max_tokens=384,temperature=1,top_p=1,stream=True,
            extra_body={'top_k':-1,'return_token_ids':True})
        try:
            for chunk in stream:
                if chunk.choices:
                    value=chunk.choices[0].model_dump()
                    pieces.append(value)
                    ids.extend(value.get('token_ids') or [])
                    if ids:break
        finally:stream.close()
        assert ids,'no native tokens before disconnect'
        # Let the host observe the disconnection; subsequent request forces cleanup.
        time.sleep(.5)
        new=traces()-before;assert len(new)==1
        cancelled=next(iter(new))
        recovery=generate(('Say hello in one sentence.','marked'))
        report['outputs'].append(recovery)
        selected,events=selections(cancelled)
        assert selected[:len(ids)]==ids and len(selected)<384
        report['disconnect']={'received_chunks':pieces,'received_token_ids':ids,
            'selected_tokens':len(selected),'cap':384,'journal':cancelled.name}
        report['checks']['disconnect_stops_before_cap_and_worker_recovers']=True
        # Ensure cancellation is stable after another scheduling cycle.
        count=len(selected); time.sleep(.3)
        assert len(selections(cancelled)[0])==count
        report['status']='pass'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        client.close();anthropic.close();save()
    return 0


if __name__=='__main__':raise SystemExit(main())
