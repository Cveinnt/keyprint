import json
import math
import pytest

from audit_paced_study import digest
from paced_regimes import entropy_bin, inspect, LABELS
from paced_source_session import integer_distribution


@pytest.mark.parametrize('value,index', [(0,0),(.1,1),(.5,2),(1,3),(2,4),(9,4)])
def test_boundaries(value,index):
    assert entropy_bin(value) == LABELS[index]


@pytest.mark.parametrize('value', [-1, math.inf, math.nan])
def test_bad_entropy(value):
    with pytest.raises(ValueError): entropy_bin(value)


def fixture(tmp_path, condition='marked', actual=(.6,.4)):
    d=integer_distribution(actual)
    events=[{'phase':'prepared','index':0,'token_ids':[1,2],
             'base_hex':[float(.5).hex()]*2,
             'distribution':{'weights':d.weights,'total':d.total},
             'decision':{'mode':'full_mark','protected_token_ids':[],
                         'boundary_freezes':3,'executed_layers':30}},
            {'phase':'committed','index':0,'token_id':2}]
    path=tmp_path/'journal.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    row={'review_id':'test','condition':condition,'journal_sha256':digest(path),
         'committed_token_ids':[2],'completion':'eos'}
    return path,row


def test_all_counters_and_metrics(tmp_path):
    path,row=fixture(tmp_path)
    result=inspect(path,row,{2})
    b=result['bins']['[1,2)']
    assert b['steps']==b['eos_selected']==b['steps_with_freezes']==1
    assert b['boundary_freezes']==3 and b['executed_layers']==30
    assert b['kl_sum']==pytest.approx(.6*math.log(1.2)+.4*math.log(.8))
    assert b['tv_sum']==pytest.approx(.1)
    assert sum(v['steps'] for v in result['bins'].values())==1


def test_ordinary_is_actual_not_counterfactual(tmp_path):
    path,row=fixture(tmp_path,'ordinary',(.5,.5))
    assert inspect(path,row,{2})['sums']['kl_marked_to_base']==0
    path,row=fixture(tmp_path,'ordinary')
    with pytest.raises(ValueError,match='Ordinary distribution'): inspect(path,row,{2})


def test_hash_and_incomplete_rejected(tmp_path):
    path,row=fixture(tmp_path)
    row['committed_token_ids']=[1]
    with pytest.raises(ValueError,match='Incomplete'): inspect(path,row,{2})
    path.write_text('')
    with pytest.raises(ValueError,match='hash'): inspect(path,row,{2})


def test_eos_not_optional(tmp_path):
    path,row=fixture(tmp_path)
    with pytest.raises(ValueError,match='EOS'): inspect(path,row,{3})
