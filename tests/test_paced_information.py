from fractions import Fraction
import itertools
import json
import math

import pytest

from paced_information import metrics, same_distribution, summarize, inspect_path, IntegerDistribution as D
from paced_source_session import PacedProfile, Config, partition, integer_distribution
from audit_paced_study import digest


def test_metrics_against_independent_fraction_formula():
    p,q=D((7,2,1),10),D((12,6,2),20)
    actual=metrics(p,q,1)
    pf=[Fraction(w,p.total) for w in p.weights];qf=[Fraction(w,q.total) for w in q.weights]
    assert actual['selected_log_ratio']==pytest.approx(math.log(1.5))
    assert actual['kl_marked_to_base']==pytest.approx(sum(float(b)*math.log(float(b/a)) for a,b in zip(pf,qf)))
    assert actual['kl_base_to_marked']==pytest.approx(sum(float(a)*math.log(float(a/b)) for a,b in zip(pf,qf)))
    assert actual['total_variation']==pytest.approx(float(sum(abs(a-b) for a,b in zip(pf,qf))/2))


def test_finite_adaptive_likelihood_identity():
    ordinary_total=marked_total=weighted_ratio=expected_log=conditional_kl=0.
    for path in itertools.product(range(2),repeat=3):
        pp=qq=1.;llr=0.;kls=0.
        for i,chosen in enumerate(path):
            p=D((3,1),4) if i and path[i-1] else D((1,1),2)
            q=D((5,3),8) if p.total==4 else D((3,5),8)
            m=metrics(p,q,chosen);kls+=m['kl_marked_to_base'];llr+=m['selected_log_ratio']
            pp*=p.weights[chosen]/p.total;qq*=q.weights[chosen]/q.total
        ordinary_total+=pp;marked_total+=qq;weighted_ratio+=pp*math.exp(llr)
        expected_log+=qq*llr;conditional_kl+=qq*kls
    assert ordinary_total==marked_total==pytest.approx(1)
    assert weighted_ratio==pytest.approx(1)
    assert expected_log==pytest.approx(conditional_kl)


def test_identity_and_near_identity_are_stable():
    p=D((10**100,10**100),2*10**100)
    assert metrics(p,p,0)['selected_log_ratio']==0
    q=D((10**100+1,10**100-1),2*10**100)
    assert metrics(p,q,0)['selected_log_ratio']>0
    assert same_distribution(D((1,1),2),D((3,3),6))


@pytest.mark.parametrize('p,q,index',[(D((0,1),1),D((1,1),2),0),
    (D((1,1),3),D((1,1),2),0),(D((True,1),2),D((1,1),2),0),
    (D((1,1),2),D((1,9),10),0),(D((1,1),2),D((1,),1),0),
    (D((1,1),2),D((1,1),2),True),(D((1,1),2),D((1,1),2),2)])
def test_invalid_distributions_and_indexes_fail(p,q,index):
    with pytest.raises(ValueError): metrics(p,q,index)


def test_ordinary_path_uses_marked_counterfactual_and_eos(tmp_path):
    profile=PacedProfile([b'a',b'b',None],tokenizer_identity='test',eos_ids={2},config=Config(layers=2))
    key=b'k'*32;raw=(.25,.25,.5);base=integer_distribution(raw)
    q,_=partition(raw,[0,1,2],profile,key,(),frozenset())
    path=tmp_path/'journal.jsonl'
    events=[{'phase':'prepared','index':0,'token_ids':[0,1,2],
        'base_hex':[v.hex() for v in raw], 'decision':{'mode':'full_mark','protected_token_ids':[]},
        'distribution':{'weights':base.weights,'total':base.total}},
        {'phase':'committed','index':0,'token_id':2}]
    path.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    row=dict(review_id='x',case='c',key_slot=0,condition='ordinary',journal_sha256=digest(path),
             committed_token_ids=[2],completion='eos')
    result=inspect_path(path,row,profile,key)
    assert result['sums']==metrics(base,q,2)
    assert result['sums']['selected_log_ratio']==0  # Exact EOS mass preservation.
    assert result['sums']['kl_marked_to_base']>0
    events[0]['distribution']={'weights':q.weights,'total':q.total}
    path.write_text('\n'.join(json.dumps(e) for e in events)+'\n');row['journal_sha256']=digest(path)
    with pytest.raises(ValueError,match='differs from generation'):inspect_path(path,row,profile,key)


def test_complete_cohort_required():
    with pytest.raises(ValueError):summarize([])
