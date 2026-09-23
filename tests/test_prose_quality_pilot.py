import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('prose_pilot',Path(__file__).parents[1]/'tools/prose_quality_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
PLAN={'cases':[{'id':'one'}],'repeats':1}

def row(condition,**extra):
    return dict(case='one',repetition=0,condition=condition,review_id=condition,completion='eos',**extra)

def label(condition,value):
    return dict(review_id=condition,task_pass=value,reason='Frozen rubric rating')

def test_missing_reviews_never_look_like_zero_failure():
    result=pilot.summarize(PLAN,[row('ordinary'),row('marked')],[label('ordinary',True)])
    assert not result['complete']
    assert result['marked_minus_ordinary_failure_rate'] is None
    assert result['conditions']['marked']['unreviewed']==1

def test_direction_and_discordance_are_not_global_acceptance():
    result=pilot.summarize(PLAN,[row('ordinary'),row('marked')],[label('ordinary',True),label('marked',False)])
    assert result['complete'] and result['marked_minus_ordinary_failure_rate']==1
    assert result['paired_outcomes']=={'ordinary_only_pass':1}
    assert result['effect']=='unresolved' and result['quality_acceptance'] is False

def test_runtime_failures_and_capped_text_cannot_be_approved():
    capped=row('ordinary');capped['completion']='length'
    result=pilot.summarize(PLAN,[capped,row('marked',error_type='RuntimeError')],[label('ordinary',True)])
    assert result['complete'] and result['paired_outcomes']=={'both_fail':1}
    assert result['conditions']['marked']['runtime_errors']==1

@pytest.mark.parametrize('rows,labels',[
    ([row('ordinary'),row('ordinary')],[]),
    ([row('ordinary')],[label('missing',True)]),
    ([row('ordinary')],[label('ordinary',True),label('ordinary',False)]),
    ([row('ordinary')],[label('ordinary','true')]),
])
def test_bad_join_rejected(rows,labels):
    with pytest.raises(ValueError):pilot.summarize(PLAN,rows,labels)
