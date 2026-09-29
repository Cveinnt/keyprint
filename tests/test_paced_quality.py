import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import summarize_paced_quality as quality
from test_paced_cohort import fixture


def ratings_for(root):
    review=json.loads((root/'private/blind-review.json').read_text())
    ratings=[]
    for row in review:
        success=not(row.get('error_type') or row.get('decode_error'))
        ratings.append({'review_id':row['review_id'],
            'fact_checks':[success for _ in row['rubric']['essential_facts']],
            'no_unsupported_claims':success,'language_pass':success,'format_pass':success,
            'uncertain_fields':[],'reason':'Synthetic fixture rating, not an actual output judgment.'})
    return review,ratings


def write_ratings(tmp_path,ratings):
    path=tmp_path/'ratings-input.json';path.write_text(json.dumps(ratings,indent=2)+'\n');return path


def test_freeze_does_not_join_conditions_or_compute_counts(tmp_path,monkeypatch):
    root,original=fixture(tmp_path);_,ratings=ratings_for(root)
    monkeypatch.setattr(quality,'summarize',lambda *a,**k:pytest.fail('No join during freeze'))
    monkeypatch.setattr(quality,'diagnostics',lambda *a,**k:pytest.fail('No signal counts during freeze'))
    path=write_ratings(tmp_path,ratings);out=tmp_path/'frozen'
    commit=quality.freeze(root,original,path,out)
    assert (out/'ratings.json').read_bytes()==path.read_bytes()
    assert 'independent human ratings remain separate' in commit['reviewer']
    assert not commit['quality_acceptance']
    with pytest.raises(FileExistsError):quality.freeze(root,original,path,out)


def test_uncertain_sensitivity_and_all_attempts_use_original_metrics(tmp_path):
    root,original=fixture(tmp_path);_,ratings=ratings_for(root)
    ordinary=next(r for r in ratings if r['review_id']=='000000000000')
    ordinary['fact_checks']=[False];ordinary['uncertain_fields']=['fact:0']
    out=tmp_path/'frozen';quality.freeze(root,original,write_ratings(tmp_path,ratings),out)
    result=quality.summarize_frozen(root,original,out)
    assert result['strict']['groups']['ordinary']['attempts']==64
    assert result['strict']['groups']['ordinary']['content']==63
    assert result['strict']['groups']['marked']['content']==64
    assert result['all_uncertain_fields_accepted']['groups']['ordinary']['content']==64
    assert result['uncertain_fields']==1
    assert result['strict']['paired']['content']['marked_only']==1
    assert not result['quality_acceptance'] and not result['launch_ready'] and not result['detector_calibrated']


@pytest.mark.parametrize('failure',['forward','decode'])
def test_failures_cannot_receive_credit_but_remain_in_denominator(tmp_path,failure):
    root,original=fixture(tmp_path,failure);_,ratings=ratings_for(root)
    out=tmp_path/'frozen';quality.freeze(root,original,write_ratings(tmp_path,ratings),out)
    result=quality.summarize_frozen(root,original,out)
    assert result['strict']['groups']['ordinary']['full_task']==63
    assert result['strict']['groups']['ordinary']['content']==63
    assert result['strict']['groups']['ordinary']['attempts']==64
    assert result['execution']['errors']+result['execution']['decode_errors']==1


@pytest.mark.parametrize('change',['missing','duplicate','blank','fact','flag_pass','flag_duplicate','reason','metadata'])
def test_invalid_or_incomplete_ratings_rejected_before_freeze(tmp_path,change):
    root,_=fixture(tmp_path);review,ratings=ratings_for(root)
    if change=='missing':ratings.pop()
    if change=='duplicate':ratings[0]=ratings[1]
    if change=='blank':ratings[0]['language_pass']=None
    if change=='fact':ratings[0]['fact_checks']=[]
    if change=='flag_pass':ratings[0]['uncertain_fields']=['language']
    if change=='flag_duplicate':ratings[0]['uncertain_fields']=['claims','claims']
    if change=='reason':ratings[0]['reason']=' '
    if change=='metadata':ratings[0]['condition']='ordinary'
    with pytest.raises(ValueError):quality.validate_ratings(review,ratings)


def test_failed_decoder_cannot_be_accepted_as_uncertain(tmp_path):
    root,_=fixture(tmp_path,'decode');review,ratings=ratings_for(root)
    failed=next(r for r in ratings if r['review_id']=='000000000000')
    failed['uncertain_fields']=['language']
    with pytest.raises(ValueError):quality.validate_ratings(review,ratings)


@pytest.mark.parametrize('change',['ratings','helper','review'])
def test_post_freeze_changes_are_rejected(tmp_path,monkeypatch,change):
    root,original=fixture(tmp_path);_,ratings=ratings_for(root)
    out=tmp_path/'frozen';quality.freeze(root,original,write_ratings(tmp_path,ratings),out)
    if change=='ratings':(out/'ratings.json').write_text((out/'ratings.json').read_text()+' ')
    if change=='helper':monkeypatch.setattr(quality,'helper_hashes',lambda:{'changed':'hash'})
    if change=='review':
        p=root/'private/blind-review.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):quality.summarize_frozen(root,original,out)


def test_summary_requires_prior_commitment(tmp_path):
    root,original=fixture(tmp_path)
    with pytest.raises(FileNotFoundError):quality.summarize_frozen(root,original,tmp_path/'not-frozen')
