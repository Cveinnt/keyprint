import json
from types import SimpleNamespace
import pytest

from audit_paced_study import digest
from compare_centered_score import inspect


def fixture(tmp_path):
    calls=[]
    def bits(key,context,label):
        calls.append((context,label))
        return (0,0) if label==b'a' else (1,1)
    profile=SimpleNamespace(classes={1:b'a',2:b'b',3:None},
                            config=SimpleNamespace(layers=2,history=1),eos_ids={3},
                            bits=bits,label=lambda token: {1:b'a',2:b'b',3:None}[token])
    events=[]
    for i,t in enumerate([1,1,3]):
        events.extend([{'phase':'prepared','index':i,'token_ids':[1,2,3],
                        'base_hex':[float(.25).hex(),float(.25).hex(),float(.5).hex()],
                        'distribution':{'weights':[1,1,2],'total':4},
                        'decision':{'mode':'full_mark','protected_token_ids':[]}},
                       {'phase':'committed','index':i,'token_id':t}])
    path=tmp_path/'journal.jsonl';path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    row={'review_id':'fixture','journal_sha256':digest(path),
         'committed_token_ids':[1,1,3],'completion':'eos'}
    return path,row,profile,calls


def test_all_prefixes_include_eos_and_repeated_context(tmp_path):
    path,row,profile,calls=fixture(tmp_path)
    result=inspect(path,row,profile,b'key')
    assert result['steps']==3 and len(calls)==4
    assert result['sums']['old_kl']==result['sums']['old_score_lift']==0
    assert result['sums']['new_score_lift']==.25
    assert result['sums']['new_tv']==.25
    assert result['sums']['new_kl_greater_steps']==2


def test_changed_hash_rejected(tmp_path):
    path,row,profile,_=fixture(tmp_path);path.write_text('')
    with pytest.raises(ValueError,match='Journal changed'): inspect(path,row,profile,b'key')


def test_incomplete_path_rejected(tmp_path):
    path,row,profile,_=fixture(tmp_path);row['committed_token_ids']=[1,3]
    with pytest.raises(ValueError,match='Incomplete'): inspect(path,row,profile,b'key')


def test_protected_mode_not_silently_reused(tmp_path):
    path,row,profile,_=fixture(tmp_path)
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events[0]['decision']['protected_token_ids']=[1]
    path.write_text(''.join(json.dumps(e)+'\n' for e in events));row['journal_sha256']=digest(path)
    with pytest.raises(ValueError,match='general-purpose'): inspect(path,row,profile,b'key')
