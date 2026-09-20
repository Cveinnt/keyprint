import copy
import importlib.util
from pathlib import Path

import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Profile, Config


@pytest.fixture
def modules(monkeypatch):
    tools=Path(__file__).parents[1]/'tools';monkeypatch.syspath_prepend(str(tools))
    values=[]
    for name in ('audit_prompt_confirmation','prompt_confirmation_selection','surrogate_likelihood'):
        spec=importlib.util.spec_from_file_location(name+'_test',tools/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);values.append(module)
    return values


def test_graph_audit_rejects_history_connected_selection(modules):
    audit,selector,_=modules
    rows=[{'instruction':f'task {i}','context':'','response':(f'answer{i} '*110),'category':'a'} for i in range(4)]
    tasks,_=selector.select(rows,{0},per_category=2,categories=('a',))
    plan={'excluded_source_indices':[0],'tasks':tasks}
    assert audit.validate_groups(rows,plan)['selected_groups']==2
    corrupted=copy.deepcopy(rows)
    corrupted[tasks[0]['source_index']]['context']='bridge'
    corrupted[0]['context']='bridge'
    with pytest.raises(AssertionError):audit.validate_groups(corrupted,plan)


def example(scorer):
    profile=Profile([None,b'a',b'b',b'a'],tokenizer_identity='audit-test',config=Config(history=1,layers=3,max_steps=32))
    data={'token_ids':[1,2,1,2], 'heads':[{'support':[1,2,3],'probabilities':[.2,.3,.5]} for _ in range(4)]}
    key=bytes(range(32))
    return profile,key,data,scorer.score(profile,key,data)


def test_every_retained_term_is_independently_reconciled(modules):
    audit,_,scorer=modules
    profile,key,data,record=example(scorer)
    checked=audit.check_score(profile,key,data,record)
    assert checked['terms']==4 and checked['transforms']==3 and checked['score_error']<1e-12


@pytest.mark.parametrize('damage',['normalization','token','reason','base','aggregate','flag'])
def test_corrupt_model_probability_or_score_cannot_pass_audit(modules,damage):
    audit,_,scorer=modules
    profile,key,data,record=example(scorer)
    if damage=='normalization':data['heads'][0]['probabilities'][0]+=.1
    elif damage=='token':record['terms'][0]['token']=2
    elif damage=='reason':record['terms'][0]['reason']='repeated_context'
    elif damage=='base':record['terms'][0]['base']=.99
    elif damage=='aggregate':record['working_log_ratio']+=1
    else:record['flagged']=not record['flagged']
    with pytest.raises(AssertionError):audit.check_score(profile,key,data,record)
