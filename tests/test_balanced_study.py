"""Synthetic callback receipts test audit plumbing; never model/quality evidence."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import pytest

from audit_paced_study import digest
from balanced_inference import generate,audit_output,save
from balanced_source_session import BalancedProfile,Config
from balanced_study import audit,blind_packet,totals,validate_preflight
from freeze_balanced_evaluation import freeze as freeze_bundle
from run_balanced_evaluation import make_plan,score_attempt
from summarize_balanced_quality import freeze as freeze_ratings,summarize_frozen

SOURCE=Path(__file__).parents[1]/'research/balanced-evaluation/cases.json'
KEYS=[bytes([i])*32 for i in range(4)]


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    base=tmp_path_factory.mktemp('balanced-fixture');bundle=base/'bundle'
    prior=base/'prior.json';prior.write_text('[]');freeze_bundle(SOURCE,[prior],bundle)
    root=base/'study';public=root/'public';private=root/'private'
    public.mkdir(parents=True);private.mkdir()
    reference={'model_assets':{'fixture':'not-a-model'},'dependencies':{}}
    original={'sdk_source_sha256':{'fixture':'not-a-model'}}
    binding=SimpleNamespace(pieces=(b'a',b' b',None),eos_ids={2})
    profile=BalancedProfile(binding.pieces,tokenizer_identity='callback-fixture-not-native',eos_ids=[2],config=Config(max_steps=1024))
    preflight=dict(reference,**original,profile_sha256=profile.digest)
    plan=make_plan(bundle,reference,original,execute=True,preflight=preflight)
    save(public/'plan.json',plan)
    save(public/'identity.json',{'profile_sha256':profile.digest,'binding_sha256':'fixture'})
    cases=json.loads((bundle/'private/cases.json').read_text());rubrics=json.loads((bundle/'private/rubrics.json').read_text())
    save(private/'cases.json',cases);save(private/'rubrics.json',rubrics)
    outcomes=[]
    for i,item in enumerate(plan['schedule']):
        row=dict(item,review_id=f'{i:012x}');calls=[]
        def forward(ids):
            calls.append(ids)
            return np.array([[2.,1.,-20.]] if len(calls)==1 else [[-np.inf,-np.inf,0.]],np.float32)
        result=generate(profile=profile,binding=binding,key=KEYS[row['key_slot']],condition=row['condition'],
                        prompt_ids=[0,1],forward=forward,decode=lambda ids:''.join('a' if i==0 else ' b' for i in ids),
                        random_bits=lambda n:0,output=private/row['review_id'],max_tokens=768)
        row.update(result);row['result_sha256']=digest(private/row['review_id']/'result.json')
        row['audit']=audit_output(private/row['review_id'],profile,KEYS[row['key_slot']],binding=binding)
        row['raw_counts']=score_attempt(row,profile,KEYS);outcomes.append(row)
    save(private/'runs.json',outcomes);save(private/'blind-review.json',blind_packet(outcomes,cases,rubrics))
    finished=dict(schema='keyprint.balanced-evaluation-results.v1',**totals(outcomes),
                  plan_sha256=digest(public/'plan.json'),runs_sha256=digest(private/'runs.json'),
                  identity_sha256=digest(public/'identity.json'),review_sha256=digest(private/'blind-review.json'),
                  quality_acceptance=False,detector_calibrated=False,launch_ready=False)
    save(public/'results.json',finished)
    return base


@pytest.fixture
def study(template,tmp_path):
    shutil.copytree(template/'study',tmp_path/'study');shutil.copytree(template/'bundle',tmp_path/'bundle')
    return tmp_path/'study',tmp_path/'bundle'


def refresh(root,name):
    path=root/'public/results.json';d=json.loads(path.read_text())
    d[name]=digest(root/('private/runs.json' if name=='runs_sha256' else 'private/blind-review.json'))
    save(path,d)


def test_complete_callback_cohort_and_blind_schema(study):
    root,bundle=study;checked=audit(root,bundle)
    assert checked['attempts']==128 and checked['audited_runs']==128 and checked['committed_tokens']==256
    assert not checked['native_heads_replayed'] and not checked['quality_acceptance']
    rows=json.loads((root/'private/blind-review.json').read_text())
    assert all(not {'condition','key_slot','raw_counts'}&set(r) for r in rows)


@pytest.mark.parametrize('change',['drop','duplicate','reorder','score','audit'])
def test_rows_cannot_drop_or_forge_checks_after_hash_refresh(study,change):
    root,bundle=study;p=root/'private/runs.json';rows=json.loads(p.read_text())
    if change=='drop':rows.pop()
    if change=='duplicate':rows[-1]=rows[0]
    if change=='reorder':rows[0],rows[1]=rows[1],rows[0]
    if change=='score':rows[0]['raw_counts'][0]['trials']=999
    if change=='audit':del rows[0]['audit']
    save(p,rows);refresh(root,'runs_sha256')
    with pytest.raises(ValueError):audit(root,bundle)


@pytest.mark.parametrize('change',['text','condition','drop'])
def test_blind_packet_cannot_hide_or_alter_outputs(study,change):
    root,bundle=study;p=root/'private/blind-review.json';rows=json.loads(p.read_text())
    if change=='text':rows[0]['text']='altered'
    if change=='condition':rows[0]['condition']='ordinary'
    if change=='drop':rows.pop()
    save(p,rows);refresh(root,'review_sha256')
    with pytest.raises(ValueError):audit(root,bundle)


def test_claim_cannot_upgrade_execution_to_acceptance(study):
    root,bundle=study;p=root/'public/results.json';d=json.loads(p.read_text());d['quality_acceptance']=True;save(p,d)
    with pytest.raises(ValueError):audit(root,bundle)


def test_failed_capped_attempts_are_unavailable_not_negative_controls():
    for extra in [{'completion':'cap'},{'completion':'incomplete','error_type':'OSError'},
                  {'completion':'eos','decode_error':{}},{'completion':'eos','audit_error':{}}]:
        scores=score_attempt(dict(key_slot=0,**extra),None,KEYS)
        assert len(scores)==2 and all('unavailable' in s and 'ones' not in s for s in scores)


def test_missing_preflight_and_execution_bypass_rejected(tmp_path):
    with pytest.raises(ValueError,match='Completed'):validate_preflight(tmp_path,tmp_path)


@pytest.mark.parametrize('change',[None,'guard','helpers','audit','key','settings'])
def test_preflight_completion_receipts(template,tmp_path,change):
    # Reuse synthetic callback outputs solely to exercise receipt validation.
    from preflight_balanced import CASES
    from balanced_source_session import policy_spec
    root=tmp_path/'prefix';shutil.copytree(template/'study',root)
    public=root/'public';private=root/'private';guard=tmp_path/'guard';guard.mkdir()
    save(guard/'result.json',dict(status='completed',exit_code=0,cleanup_verified=change!='guard'))
    schedule=[dict(case=c['id'],key_slot=i,condition=arm) for i,c in enumerate(CASES)
              for arm in ('ordinary','marked')]
    rows=json.loads((private/'runs.json').read_text())[:4]
    rows[2],rows[3]=rows[3],rows[2]  # Study alternates order; preflight does not.
    for r,a in zip(rows,schedule,strict=True):r.update(a)
    save(private/'cases.json',CASES)
    for i in range(2):(private/f'key-{i}').write_bytes(KEYS[i])
    names=('preflight_balanced.py','balanced_inference.py','balanced_source_session.py',
           'balanced_score.py','centered_score.py','paced_source_session.py',
           'paced_integer_kernel.py','replay_wide_mlx.py','memory_watchdog.py','mlx_guarded_worker.py')
    plan=dict(schema='keyprint.balanced-native-preflight-plan.v1',stage='preflight',schedule=schedule,
              policy=policy_spec(),scripts_sha256={n:digest(SOURCE.parents[1].parent/'tools'/n) for n in names},
              cases_sha256=digest(private/'cases.json'),key_sha256=[hashlib.sha256(k).hexdigest() for k in KEYS[:2]],
              settings=dict(max_tokens=16,temperature=.7,top_k=100,profile_max_steps=1024,enable_thinking=False),
              model_assets={},dependencies={},sdk_source_sha256={})
    if change=='helpers':plan['scripts_sha256']={}
    if change=='audit':del rows[0]['audit']['verified_draws']
    if change=='key':(private/'key-0').write_bytes(b'x'*32)
    if change=='settings':plan['settings']['temperature']=1
    save(public/'plan.json',plan);save(private/'runs.json',rows)
    save(public/'results.json',dict(schema='keyprint.balanced-native-preflight-results.v1',attempts=4,failures=0,
        audited_runs=4,committed_tokens=8,caps=0,eos=4,plan_sha256=digest(public/'plan.json'),
        runs_sha256=digest(private/'runs.json'),identity_sha256=digest(public/'identity.json'),
        quality_acceptance=False,detector_calibrated=False,launch_ready=False))
    if change:
        with pytest.raises(ValueError):validate_preflight(root,guard)
    else:assert validate_preflight(root,guard)['profile_sha256']==rows[0]['profile_sha256']


def test_plan_only_cannot_be_a_completed_run(study):
    root,bundle=study;p=root/'public/plan.json';plan=json.loads(p.read_text())
    plan['model_load_authorized_in_this_invocation']=False;save(p,plan)
    results=json.loads((root/'public/results.json').read_text());results['plan_sha256']=digest(p);save(root/'public/results.json',results)
    with pytest.raises(ValueError):audit(root,bundle)


def ratings(root):
    # Artificial booleans exercise aggregation, not judgments of fixture text.
    return [dict(review_id=r['review_id'],fact_checks=[True]*6,no_unsupported_claims=True,
                 language_pass=True,format_pass=True,uncertain_fields=[],reason='Synthetic test rating, not a quality judgment')
            for r in json.loads((root/'private/blind-review.json').read_text())]


def test_frozen_ratings_language_counts_and_no_rerating(study,tmp_path):
    root,bundle=study;path=tmp_path/'ratings.json';values=ratings(root)
    values[0]['language_pass']=False;values[0]['uncertain_fields']=['language']
    save(path,values);frozen=tmp_path/'frozen';freeze_ratings(root,bundle,path,frozen)
    result=summarize_frozen(root,bundle,frozen)
    assert result['strict']['groups']['ordinary']['full_task']==63
    assert result['all_uncertain_fields_accepted']['groups']['ordinary']['full_task']==64
    assert result['strict_by_language']['English']['ordinary']['language']==15
    assert all(v['ordinary']['attempts']==16 and v['marked']['attempts']==16 for v in result['strict_by_language'].values())
    assert not result['quality_acceptance']
    with pytest.raises(FileExistsError):freeze_ratings(root,bundle,path,frozen)
    (frozen/'ratings.json').write_text((frozen/'ratings.json').read_text()+' ')
    with pytest.raises(ValueError,match='Frozen'):summarize_frozen(root,bundle,frozen)


def test_missing_and_invalid_ratings_cannot_freeze(study,tmp_path):
    root,bundle=study;path=tmp_path/'ratings.json';values=ratings(root)
    values.pop();save(path,values)
    with pytest.raises(ValueError):freeze_ratings(root,bundle,path,tmp_path/'frozen')
    assert not (tmp_path/'frozen').exists()
