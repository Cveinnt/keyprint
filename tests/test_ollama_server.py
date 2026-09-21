"""Ollama client protocol over Keyprint; never claims an Ollama runtime hook."""
from concurrent.futures import ThreadPoolExecutor
import pytest

ollama=pytest.importorskip('ollama')
from fastapi.testclient import TestClient
from keyprint.server import create_app
from test_server import Model, TOKEN, HEADERS

BODY={'model':'keyprint','messages':[{'role':'user','content':'hello'}],
      'options':{'num_predict':8},'stream':False}


@pytest.mark.parametrize('mode',['chat','generate'])
def test_official_client_serialization_and_exact_replay(tmp_path,mode):
    model=Model()
    with TestClient(create_app(lambda:model,api_key=TOKEN,output=tmp_path/'http')) as transport:
        client=ollama.Client(host='http://testserver',headers={**HEADERS,'Idempotency-Key':'same'})
        client._client.close()
        client._client=transport
        transport.headers.update({**HEADERS,'Idempotency-Key':'same'})
        method=getattr(client,mode)
        args={'messages':BODY['messages']} if mode=='chat' else {'prompt':'hello'}
        first=method(model='keyprint',options={'num_predict':8},stream=False,**args)
        repeated=method(model='keyprint',options={'num_predict':8},stream=False,**args)
        assert first.model_dump()==repeated.model_dump()
        assert (first.message.content if mode=='chat' else first.response)=='hello'
        assert first.done and first.done_reason=='stop'
        assert first.prompt_eval_count==4 and first.eval_count==1
        assert first.model == 'keyprint'
        assert model.calls==['hello']


@pytest.mark.parametrize('change',[
 {'model':'llama3.2'},{'stream':True},{'stream':0},{'think':True},{'keep_alive':'1h'},
 {'format':'json'},{'tools':[{'type':'function'}]}, {'options':{'num_predict':-1}},
 {'options':{'num_predict':True}}, {'options':{'num_predict':1025}},
 {'options':{'num_predict':8,'temperature':.5}}, {'options':{}},
 {'messages':[{'role':'system','content':'hello'}]},
 {'messages':BODY['messages']*2}, {'messages':[{'role':'user','content':'hi','images':['data']}]},
])
def test_unsupported_options_rejected_before_inference(tmp_path,change):
    model=Model()
    with TestClient(create_app(lambda:model,api_key=TOKEN,output=tmp_path/'http')) as client:
        result=client.post('/api/chat',json={**BODY,**change},headers=HEADERS)
        assert result.status_code==400 and isinstance(result.json()['error'],str)
        assert model.calls==[]


def test_auth_errors_and_cross_protocol_key_conflicts(tmp_path):
    model=Model()
    with TestClient(create_app(lambda:model,api_key=TOKEN,output=tmp_path/'http')) as client:
        assert client.post('/api/chat',json=BODY).status_code==401
        assert isinstance(client.post('/api/chat',json=BODY).json()['error'],str)
        headers={**HEADERS,'Idempotency-Key':'shared'}
        response=client.post('/api/chat',json=BODY,headers=headers)
        assert response.status_code==200
        assert response.json()['keyprint']['ollama_runtime'] is False
        assert client.post('/api/generate',json={'model':'keyprint','prompt':'hello','options':BODY['options']},headers=headers).status_code==409
        assert model.calls==['hello']


def test_cancel_shared_worker_replay_and_recovery(tmp_path):
    model=Model(block=True)
    with TestClient(create_app(lambda:model,api_key=TOKEN,output=tmp_path/'http')) as client:
        headers={**HEADERS,'Idempotency-Key':'cancel'}
        with ThreadPoolExecutor(max_workers=1) as pool:
            first=pool.submit(client.post,'/api/chat',json=BODY,headers=headers)
            try:
                assert model.started.wait(5)
                assert client.post('/api/chat',json=BODY,headers=headers).status_code==409
                assert client.post('/v1/keyprint/cancel',headers=headers).status_code==202
            finally:model.release.set()
            cancelled=first.result(timeout=10)
        assert cancelled.status_code==410
        assert client.post('/api/chat',json=BODY,headers=headers).json()==cancelled.json()
        assert model.calls==['hello']
        assert client.post('/api/chat',json=BODY,headers=HEADERS).status_code==200
        assert model.calls==['hello','hello']
