import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def selector():
    path=Path(__file__).parents[1]/"tools/prompt_confirmation_selection.py"
    spec=importlib.util.spec_from_file_location("confirmation_selection_test",path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def row(instruction,context,response,category="a"):
    return {"instruction":instruction,"context":context,"response":response,"category":category}


def test_exclusion_crosses_ineligible_transitive_bridge_and_normalization(selector):
    answer=" ".join(["answer"]*110)
    other=" ".join(["other"]*110)
    rows=[row("Old prompt","Anchor",answer),row("  ＯＬＤ   PROMPT  ","Bridge","short"),
          row("Should be excluded","  bridge ",other),row("New prompt","",other+" different"),
          row("New prompt","","a duplicate group answer"),row("Second new","",answer+" distinct")]
    tasks,meta=selector.select(rows,{0},per_category=2,categories=("a",))
    assert {r["source_index"] for r in tasks}=={3,5}
    assert any(r["group_indices"]==[3,4] for r in tasks)
    assert meta["excluded_groups"]==1
    again,_=selector.select(rows,{0},per_category=2,categories=("a",))
    assert tasks==again


def test_no_replacement_when_new_category_groups_insufficient(selector):
    rows=[row("old","","old "*110),row("new","","new "*110)]
    with pytest.raises(ValueError,match="Insufficient"):
        selector.select(rows,{0},per_category=2,categories=("a",))


@pytest.mark.parametrize("indices", [set(),{-1},{True},{99}])
def test_bad_history_rejected(selector,indices):
    with pytest.raises(ValueError,match="indices"):
        selector.select([row("one","","one "*110)],indices)


def test_nested_manifest_indices_and_invalid_types(selector):
    assert list(selector.source_indices({"tasks":[{"source_index":4}],"nested":{"source_index":8}}))==[4,8]
    with pytest.raises(ValueError,match="integer"):
        list(selector.source_indices({"source_index":True}))
