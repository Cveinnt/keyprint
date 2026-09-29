from fractions import Fraction as F
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from balanced_source_session import (
    BalancedProfile, BalancedSourceSession, Config, SourceRequest, SamplingFailure,
    replay_scores, policy_spec, digest_spec)
from paced_source_session import PacedProfile, PacedSourceSession
from keyprint._engine.research.keyprint_exact_categorical_v2 import integer_distribution
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import replay_events

KEY=bytes(range(32))


def profile(pieces=(b'a',b'b',None),**kw):
    return BalancedProfile(pieces,tokenizer_identity='balanced-fixture',eos_ids=[len(pieces)-1],**kw)


def probabilities(step):
    return {i:F(w,step.distribution.total) for i,w in zip(step.token_ids,step.distribution.weights)}


def ordinary(q):
    d=integer_distribution(tuple(float(v) for v in q))
    return {i:F(w,d.total) for i,w in enumerate(d.weights) if w}


def choose(s,p,token):
    index=p.token_ids.index(token);point=sum(p.distribution.weights[:index])
    draw=s.draw(p,lambda count:point)
    assert draw.token_index==token
    return s.commit(p,token)


def test_distinct_domain_and_mutual_profile_rejection():
    b=profile();p=PacedProfile((b'a',b'b',None),tokenizer_identity='balanced-fixture',eos_ids=[2])
    assert b.digest!=p.digest and b.bits(KEY,(),b'a')!=p.bits(KEY,(),b'a')
    with pytest.raises(TypeError):BalancedSourceSession(p,KEY,condition='marked')
    with pytest.raises(TypeError):PacedSourceSession(b,KEY,condition='marked')
    with pytest.raises(ValueError):BalancedProfile((b'a',b'eos'),tokenizer_identity='x',eos_ids=[1])


@pytest.mark.parametrize('key',[KEY,*[bytes([i])*32 for i in range(4)]])
def test_exact_exclusions_group_ratio_and_bounds(key):
    s=BalancedSourceSession(profile((b'a',b' a',b'b',b' \n',None)),key,condition='marked')
    q=np.array([.13,.26,.2,.19,.22]);step=s.prepare(q)
    p,r=ordinary(q),probabilities(step)
    assert sum(r.values())==1 and all(p[i]/2<=r[i]<=3*p[i]/2 for i in p)
    assert r[3]==p[3] and r[4]==p[4] and r[0]/r[1]==p[0]/p[1]
    assert s.last_decision['score_adjustments']==1 and s.last_decision['score_layers']==30
    receipt=s.source_receipt()
    assert receipt['source_policy_sha256']==digest_spec(policy_spec())
    assert receipt['profile_sha256']==s.profile.digest and not receipt['quality_acceptance']


def test_source_overlap_protection_and_startup_remain_exact():
    prefix=b'abcdefghijklmnopqrstuvwx';pieces=(prefix,b'A',b' A',b'B',b'C',b'D',None)
    s=BalancedSourceSession(profile(pieces),KEY,condition='marked',
                            request=SourceRequest('proofread',(prefix+b'A'+prefix+b'B').decode()))
    first=s.prepare(np.array([1.,0.,0.,0.,0.,0.,0.]));assert s.last_decision['mode']=='startup_ordinary'
    choose(s,first,0)
    q=np.array([0.,.1,.1,.2,.2,.3,.1]);step=s.prepare(q)
    assert set(s.last_decision['protected_token_ids'])=={1,2,3}
    assert s.last_decision['alignment_occurrences']==2
    p,r=ordinary(q),probabilities(step)
    assert all(p[i]==r[i] for i in (1,2,3,6))


def test_repeated_context_and_exact_score_replay():
    p=profile(config=Config(history=1,max_steps=8))
    s=BalancedSourceSession(p,KEY,condition='marked');events=[]
    for token in [0,0,1,2]:
        prepared=s.prepare(np.array([.4,.4,.2]))
        if len(events)==2:
            assert probabilities(prepared)==ordinary(np.array([.4,.4,.2]))
            assert not s.last_decision['partitioned']
        events.append(choose(s,prepared,token))
    replay=replay_scores(p,KEY,[0,0,1,2])
    actual=[e for e in events if e is not None and e.eligible]
    assert replay['ones']==sum(sum(e.bits) for e in actual) and replay['events']==len(actual)
    independent=replay_events(p,KEY,[0,0,1,2])
    assert replay['ones']==sum(sum(e.bits) for e in independent if e.eligible)
    with pytest.raises(RuntimeError,match='closed'):s.prepare(np.array([.4,.4,.2]))
    with pytest.raises(ValueError,match='after EOS'):replay_scores(p,KEY,[0,2,1])


def test_smallest_subnormal_support_is_retained_exactly():
    s=BalancedSourceSession(profile(),KEY,condition='marked')
    q=np.array([1.,2.**-1074,0.]);p=s.prepare(q);a,b=ordinary(q),probabilities(p)
    assert set(b)=={0,1} and all(a[i]/2<=b[i]<=3*a[i]/2 for i in a)
    choose(s,p,1)


def test_rejection_transcript_draw_cap_and_foreign_prepared():
    s=BalancedSourceSession(profile(),KEY,condition='ordinary');other=BalancedSourceSession(profile(),KEY,condition='ordinary')
    p=s.prepare(np.array([1/3]*3));foreign=other.prepare(np.array([1/3]*3))
    with pytest.raises(ValueError):s.draw(foreign,lambda _:0)
    with pytest.raises(ValueError):s.commit(p,0)
    with pytest.raises(SamplingFailure) as error:s.draw(p,lambda _:3,max_draws=2)
    assert len(error.value.transcript)==2 and error.value.callback_calls==2
    with pytest.raises(RuntimeError,match='closed'):s.draw(p,lambda _:0)


def test_owner_and_step_cap_enforced():
    s=BalancedSourceSession(profile(config=Config(max_steps=1)),KEY,condition='marked')
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError):pool.submit(s.prepare,np.array([1.,0.,0.])).result()
    choose(s,s.prepare(np.array([1.,0.,0.])),0)
    with pytest.raises(RuntimeError,match='cap'):s.prepare(np.array([1.,0.,0.]))


def test_kernel_exception_closes_without_ordinary_fallback(monkeypatch):
    import balanced_source_session as module
    def fail(*args):raise ArithmeticError('fixture allocation failure')
    monkeypatch.setattr(module,'allocate',fail)
    s=BalancedSourceSession(profile(),KEY,condition='marked')
    with pytest.raises(ArithmeticError):s.prepare(np.array([.4,.4,.2]))
    with pytest.raises(RuntimeError,match='closed'):s.prepare(np.array([.4,.4,.2]))


@pytest.mark.parametrize('condition',['ordinary','marked'])
def test_single_class_identity_and_zero_entropy_draw(condition):
    s=BalancedSourceSession(profile((b'a',b' a',None)),KEY,condition=condition)
    q=np.array([.2,.6,.2]);step=s.prepare(q)
    assert probabilities(step)==ordinary(q) and not s.last_decision['partitioned']
    s.close()
    s=BalancedSourceSession(profile(),KEY,condition=condition);step=s.prepare(np.array([0.,0.,1.]))
    draw=s.draw(step,lambda _:pytest.fail('No randomness needed'))
    assert draw.transcript==() and draw.token_index==2
    s.commit(step,2)
