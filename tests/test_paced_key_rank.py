import itertools

import pytest

from paced_key_rank import compile_paths, key_rank, score_paths, summarize
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile, replay_events


def test_compiled_scores_match_independent_replay_with_fragments_and_repeats():
    profile = Profile([b'a', b'b', b' a ', b'\xe4', b'\xbd\xa0', b' ', None],
                      tokenizer_identity='test', config=Config(history=2, layers=30, max_steps=20))
    paths = [[0, 1, 0, 1, 0, 1, 5, 6], [2, 3, 4, 5, 0], [], [5, 6]]
    paths += [list(x) for x in itertools.product(range(3), repeat=3)]
    messages, indexes = compile_paths(profile, paths)
    for key in (b'a'*32, b'b'*32, bytes(range(32))):
        expected = [sum(sum(e.bits) for e in replay_events(profile, key, ids) if e.eligible) for ids in paths]
        assert score_paths(key, messages, indexes) == expected


def test_ties_are_conservative_and_threshold_exact():
    assert key_rank(5, [5]*199, 10)['random_key_rank'] == 1
    assert key_rank(6, [5]*199, 10)['rank_numerator'] == 1
    assert key_rank(6, [6]+[5]*198, 10)['at_or_below_one_percent']
    assert not key_rank(6, [6]*2+[5]*197, 10)['at_or_below_one_percent']


@pytest.mark.parametrize('observed,decoys,trials', [(True,[1],2), (1,[],2), (1,[False],2),
    (-1,[1],2),(3,[1],2),(1,[3],2),(0,[0],0),(0,[0],True)])
def test_rank_rejects_invalid_or_unavailable_counts(observed, decoys, trials):
    with pytest.raises(ValueError):
        key_rank(observed, decoys, trials)


def cohort():
    return [dict(review_id=f'{c}-{k}-{a}', case=c, key_slot=k, condition=a)
            for c in range(16) for k in range(4) for a in ('ordinary', 'marked')]


def test_unavailable_rows_remain_in_denominator():
    rows = cohort()
    rows[0]['rank'] = key_rank(6,[5]*199,10)
    result = summarize(rows)
    assert result['groups']['ordinary']['attempts'] == 64
    assert result['groups']['ordinary']['unavailable'] == 63
    assert result['groups']['marked']['available'] == 0
    assert not result['detector_calibrated'] and not result['launch_ready']


def test_partial_duplicate_and_unplanned_cohorts_rejected():
    rows = cohort()
    for bad in (rows[:-1], rows[:-1]+[rows[0]], [dict(r,key_slot=5) for r in rows]):
        with pytest.raises(ValueError):
            summarize(bad)


def test_compile_rejects_invalid_token_and_cap():
    profile = Profile([b'a'],tokenizer_identity='test',config=Config(max_steps=1))
    for paths in ([[0,0]], [[1]], [[False]]):
        with pytest.raises(ValueError):
            compile_paths(profile,paths)


def test_score_rejects_invalid_key():
    with pytest.raises(ValueError):
        score_paths(b'bad',[],[])
