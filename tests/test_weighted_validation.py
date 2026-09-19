import importlib.util
from pathlib import Path
import sys
import pytest


def load(name, monkeypatch):
    path=Path(__file__).parents[1]/'tools'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,name,module);spec.loader.exec_module(module)
    return module


def test_fresh_screen_cannot_reuse_prior_groups(monkeypatch):
    for name in ['compare_score_baselines','layer_likelihood','weighted_null','validate_null_corpus']:
        load(name,monkeypatch)
    tool=load('validate_weighted_null',monkeypatch)
    source=[{'instruction':str(i),'context':'','response':('word'+str(i)+' ')*100,'category':'qa'} for i in range(1500)]
    prior_rows,_=tool.select_rows(source,1000)
    prior={'records':prior_rows}
    fresh,_=tool.fresh_rows(source,prior)
    assert len(fresh)==500
    assert not {r['source_index'] for r in fresh} & {r['source_index'] for r in prior_rows}
    with pytest.raises(ValueError,match='selection differs'):
        tool.fresh_rows(source,{'records':prior_rows[:-1]})
