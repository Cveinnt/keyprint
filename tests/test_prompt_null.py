import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).parents[1] / 'tools'
    monkeypatch.syspath_prepend(str(directory))
    result = []
    for name in ('prompt_null_selection', 'validate_prompt_null'):
        spec = importlib.util.spec_from_file_location(name, directory / (name + '.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result.append(module)
    return result


def row(instruction, context, response, category='general_qa'):
    return dict(instruction=instruction, context=context, response=response, category=category)


def test_full_graph_exclusion_crosses_ineligible_bridge(modules):
    select, _ = modules
    rows = [row('Old', 'anchor', 'old ' * 110), row('  ＯＬＤ ', 'bridge', 'short'),
            row('blocked', ' BRIDGE ', 'blocked ' * 110),
            row('new', '', 'new ' * 110), row('new', '', 'different short answer'),
            row('another', '', 'another ' * 110, 'open_qa')]
    selected, meta = select.select(rows, {0}, count=2)
    assert {item['source_index'] for item in selected} == {3, 5}
    assert next(item for item in selected if item['source_index'] == 3)['group_indices'] == [3, 4]
    assert meta['available_eligible_groups'] == 2
    assert select.select(rows, {0}, count=2) == (selected, meta)
    assert all('prompt' not in item and 'text' not in item for item in selected)
    with pytest.raises(ValueError, match='Insufficient'):
        select.select(rows, {0}, count=3)


@pytest.mark.parametrize('indices', [set(), {-1}, {True}, {99}])
def test_invalid_history_rejected(modules, indices):
    with pytest.raises(ValueError, match='indices'):
        modules[0].select([row('p', '', 'word ' * 110)], indices, count=1)


def test_two_keys_are_one_document_and_one_hit_fails_bound(modules):
    _, runner = modules
    rows = [{'flagged': False, 'flags': [False, False]} for _ in range(500)]
    result = runner.summarize(rows)
    assert result['null_screen_passed']
    assert result['iid_only_upper_97_5_percent'] == pytest.approx(0.007350610051907794)
    rows[0] = {'flagged': True, 'flags': [True, True]}
    result = runner.summarize(rows)
    assert result['false_hits'] == 1
    assert not result['null_screen_passed']
    assert result['iid_only_upper_97_5_percent'] > .01


def test_errors_missing_rows_and_fatal_never_pass(modules):
    runner = modules[1]
    rows = [{'flagged': False} for _ in range(500)]
    for items, fatal in ((rows[:-1], None), (rows, 'fatal'), (rows[:-1] + [{'error': 'failed'}], None)):
        result = runner.summarize(items, fatal)
        assert not result['null_screen_passed']
        assert result['iid_only_upper_97_5_percent'] is None
    assert runner.summarize(rows[:-1] + [{'error': 'failed'}])['unavailable'] == 1
