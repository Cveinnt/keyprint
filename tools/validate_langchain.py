"""Actual LangChain requests against Keyprint's local GGUF worker; no hosted calls."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

# Disable opt-in external tracing even when inherited from the developer shell.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from langchain_openai import ChatOpenAI
from openai import BadRequestError
import uvicorn
from keyprint import Keyprint
from keyprint.server import create_app

CASES = [
    ('english', 'Explain in one sentence why a seed needs water.'),
    ('french', 'En français, écris une phrase demandant à Maya de garder la sauvegarde jusqu’à la vérification de la restauration.'),
    ('spanish', 'En español, escribe una frase pidiendo a Maya que espere mi aprobación antes de publicar el documento.'),
]


def main():
    if not __debug__:
        raise RuntimeError('run without -O; validation assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(mode=0o700)
    public = root/'public'; public.mkdir()
    key, token = Keyprint.new_key(), secrets.token_hex(32)
    (root/'owner.key').write_bytes(key); (root/'owner.key').chmod(0o600)
    with args.model.open('rb') as model_file:
        model_sha256 = hashlib.file_digest(model_file, 'sha256').hexdigest()
    plan = {'cases': CASES, 'max_tokens':96, 'backend':'llama-cpp', 'context_size':2048,
            'threads':2, 'hosted_calls':False, 'versions':{n:importlib.metadata.version(n)
              for n in ['langchain-openai','langchain-core','openai','llama-cpp-python']},
            'scope':'Text-only sync/async requests and replay; quality and detection not established',
            'model_sha256':model_sha256}
    (public/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    report={'status':'running','ordinary':[],'marked':[],'quality_acceptance':False}
    calls=[]
    def save():
        (public/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    def loader():
        model=Keyprint.from_llama_cpp(args.model,key=key,context_size=2048,threads=2)
        report['identity']=model.identity
        for name,prompt in CASES:
            result=model.generate(prompt,max_tokens=96,condition='ordinary',output=root/('ordinary-'+name))
            report['ordinary'].append({'id':name,'prompt':prompt,'text':result.text,'usage':result.report['usage'],
                                       'completion':result.report.get('payload',result.report)['completion']})
            save()
        generate=model.generate
        def counted(*a,**kw):
            calls.append(kw.get('condition','marked'))
            return generate(*a,**kw)
        model.generate=counted
        return model
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    app=create_app(loader,api_key=token,output=root/'http')
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        deadline=time.monotonic()+240
        while not server.started:
            if not thread.is_alive() or time.monotonic()>deadline:raise RuntimeError('local startup failed')
            time.sleep(.1)
        chat=ChatOpenAI(model='keyprint',base_url=f'http://127.0.0.1:{port}/v1',api_key=token,
                        max_tokens=96,max_retries=0,timeout=180,use_responses_api=False,temperature=None)
        for i,(name,prompt) in enumerate(CASES):
            headers={'Idempotency-Key':'langchain-'+name}
            if i==1:
                async def invoke():
                    a=await chat.ainvoke(prompt,extra_headers=headers)
                    b=await chat.ainvoke(prompt,extra_headers=headers)
                    return a,b
                first,replay=asyncio.run(invoke())
            else:
                first=chat.invoke(prompt,extra_headers=headers)
                replay=chat.invoke(prompt,extra_headers=headers)
            assert first.content and first.content==replay.content
            assert first.usage_metadata==replay.usage_metadata
            assert len(calls)==i+1
            report['marked'].append({'id':name,'prompt':prompt,'text':first.content,
                'usage':first.usage_metadata,'metadata':first.response_metadata,'replay_exact':True,
                'mode':'async' if i==1 else 'sync'})
            save();print(name,'completed with exact replay',flush=True)
        report['rejections']={}
        for name,kwargs in [('tools',{'tools':[{'type':'function','function':{'name':'noop','parameters':{'type':'object'}}}]}),
                            ('multi_turn',{})]:
            messages=[('human','Hello'),('ai','Hello'),('human','Again')] if name=='multi_turn' else 'Hello'
            try:chat.invoke(messages,extra_headers={'Idempotency-Key':'reject-'+name},**kwargs)
            except BadRequestError as exc:
                assert exc.status_code==400;report['rejections'][name]=400
            else:raise AssertionError(name+' unexpectedly accepted')
        journals=list((root/'http').rglob('journal.jsonl'))
        assert len(journals)==len(CASES) and len(calls)==len(CASES)
        saved_reports=[]
        for path in (root/'http').rglob('report.json'):
            value=json.loads(path.read_text());value=value.get('report',value)
            text=value.get('text',value.get('rendered_carriers',{}).get('visible_text'))
            saved_reports.append((text,value['usage']['completion_tokens']))
        for row in report['marked']:
            assert (row['text'],row['usage']['output_tokens']) in saved_reports
        report.update(status='pass',actual_marked_requests=len(calls),journals=len(journals),
                      exact_text_and_usage_match_private_receipts=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        save();server.should_exit=True;thread.join(180);sock.close()
        if thread.is_alive():raise RuntimeError('server failed to stop')
    return 0

if __name__=='__main__':raise SystemExit(main())
