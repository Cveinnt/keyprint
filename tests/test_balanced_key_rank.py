"""Tiny synthetic paths qualify plumbing, not detection performance."""
import pytest
from balanced_key_rank import evaluate,aggregate
from balanced_source_session import BalancedProfile,Config,replay_scores


def fixture():
    profile=BalancedProfile([b'a',b'b',b' ',None],tokenizer_identity='synthetic',eos_ids=[3],config=Config(max_steps=1024))
    keys=[i.to_bytes(32,'big') for i in range(203)]
    rows=[]
    for c in range(16):
        for k in range(4):
            for a in ('ordinary','marked'):
                tokens=[0,1,2,0,1,3]
                rows.append(dict(review_id=f'{c}-{k}-{a}',case=str(c),key_slot=k,condition=a,
                    completion='eos',committed_token_ids=tokens,
                    raw_counts=[replay_scores(profile,keys[s],tokens) for s in (k,(k+1)%4)]))
    return profile,rows,keys[:4],keys[4:]


def test_both_key_views_match_replay_and_retain_unavailable():
    profile,rows,keys,decoys=fixture();rows[0]['completion']='cap';rows[1]['decode_error']={}
    emissions=[];scored=evaluate(profile,rows,keys,decoys,emissions.append)
    assert len(emissions)==199
    assert 'rank' not in scored[0] and 'next_key_rank' not in scored[1]
    assert all(r['rank']['rank_denominator']==200 for r in scored[2:])
    cases=[dict(id=str(i),language=['English','Spanish','French','Chinese'][i//4]) for i in range(16)]
    summary=aggregate(scored,cases)
    for arm in ('ordinary','marked'):
        assert summary['groups'][arm]['attempts']==64
        assert summary['groups'][arm]['unavailable']==1
        assert summary['next_key_groups'][arm]['unavailable']==1
    assert not summary['launch_ready'] and not summary['detector_calibrated']
    assert summary['by_language']['English']['ordinary']['available']==15


@pytest.mark.parametrize('change',['count','collision','length','score','after_eos'])
def test_changed_decoys_or_score_path_rejected(change):
    profile,rows,keys,decoys=fixture()
    if change=='count':decoys.pop()
    if change=='collision':decoys[0]=keys[0]
    if change=='length':decoys[0]=b'x'
    if change=='score':rows[0]['raw_counts'][0]['ones']+=1
    if change=='after_eos':rows[0]['committed_token_ids'].append(0)
    with pytest.raises(ValueError):evaluate(profile,rows,keys,decoys)
