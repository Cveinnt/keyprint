import importlib.util
from pathlib import Path
import sys

# Tool imports use their own directory when invoked as documented scripts.
TOOLS=Path(__file__).parents[1]/"tools"
def load(name):
    spec=importlib.util.spec_from_file_location(name,TOOLS/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_confirmation_retains_failed_attempts_and_uses_frozen_threshold(monkeypatch):
    monkeypatch.setitem(sys.modules,"compare_score_baselines",load("compare_score_baselines"))
    monkeypatch.setitem(sys.modules,"validate_detection_screen",load("validate_detection_screen"))
    tool=load("validate_score_confirmation")
    rows=[{"condition":"marked","scores":{"matching":{"fixed":2},"other":{"fixed":3}}},
          {"condition":"marked","error":"RuntimeError"}]
    result=tool.summarize(rows,{"panels":{"fixed":{"threshold":2}}})['fixed']['groups']
    assert result['marked_matching']=={'attempts':2,'available':1,'unavailable':1,'above':0}
    assert result['marked_other']['above']==1
    assert result['ordinary_matching']['attempts']==0
