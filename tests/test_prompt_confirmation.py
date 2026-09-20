import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    tools=Path(__file__).parents[1]/'tools';monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location('prompt_confirmation_test',tools/'validate_prompt_confirmation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def rows():
    return [{'id':f'{i}-{c}','condition':c,'words':150,'completion':'eos','matching_flagged':c=='marked','other_flagged':False} for i in range(12) for c in ('ordinary','marked')]


def test_errors_missing_rows_and_negative_flags_cannot_pass(runner):
    source=rows();assert runner.summarize(source)['confirmation_screen_passed']
    assert not runner.summarize(source[:-1])['confirmation_screen_passed']
    source[0]['error']={'type':'Error'};assert not runner.summarize(source)['confirmation_screen_passed']
    del source[0]['error'];source[0]['other_flagged']=True
    assert not runner.summarize(source)['confirmation_screen_passed']
    source[0]['other_flagged']=False
    assert not runner.summarize(source,'fatal')['confirmation_screen_passed']


def test_short_gate_rounds_up_and_retains_caps(runner):
    source=rows();marked=[r for r in source if r['condition']=='marked']
    for r in marked[6:]:r['words']=450
    marked[0]['matching_flagged']=False;marked[1]['matching_flagged']=False
    # 10/12 overall is insufficient when only 4/6 short texts are found.
    assert not runner.summarize(source)['confirmation_screen_passed']
    marked[1]['matching_flagged']=True;marked[0]['completion']='length'
    result=runner.summarize(source)
    assert result['confirmation_screen_passed'] and result['truncated']==1
    assert result['short_marked']['required_hits']==5
    for r in marked[4:]:r['words']=450
    assert not runner.summarize(source)['confirmation_screen_passed']
