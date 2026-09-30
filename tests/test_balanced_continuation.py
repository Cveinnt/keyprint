"""Synthetic stopped-worker fixtures; no native model or quality evidence."""
import json
from pathlib import Path
import shutil
import pytest
from test_balanced_study import template
from audit_paced_study import digest
from balanced_study import read,audit
from balanced_inference import save
from balanced_continuation import inventory,prepare,verify_parent,audit_lineage,finalize


@pytest.fixture
def stopped(template,tmp_path):
    parent=tmp_path/'parent';bundle=tmp_path/'bundle';guard=tmp_path/'guard';guard.mkdir()
    shutil.copytree(template/'study',parent);shutil.copytree(template/'bundle',bundle)
    allrows=read(parent/'private/runs.json');kept=allrows[:3];interrupted=allrows[3]
    for r in allrows[4:]:shutil.rmtree(parent/'private'/r['review_id'])
    folder=parent/'private'/interrupted['review_id'];(folder/'result.json').unlink()
    journal=folder/'journal.jsonl';journal.write_text(''.join(journal.read_text().splitlines(keepends=True)[:4]))
    save(parent/'private/runs.json',kept)
    save(parent/'private/prompt-ids.json',{c['id']:[0,1] for c in read(parent/'private/cases.json')})
    (parent/'public/results.json').unlink();(parent/'private/blind-review.json').unlink()
    save(guard/'result.json',dict(status='stopped',reason='system_pressure',cleanup_verified=True,exit_code=-15))
    checkpoint=tmp_path/'checkpoint.json'
    save(checkpoint,dict(schema='keyprint.balanced-interrupted-study.v1',scheduled=128,completed_eos=3,
        completed_committed_tokens=6,interrupted_attempts=1,unstarted_attempts=124,
        plan_sha256=digest(parent/'public/plan.json'),runs_sha256=digest(parent/'private/runs.json'),
        identity_sha256=digest(parent/'public/identity.json'),supervisor_sha256=digest(guard/'result.json'),
        unfinished_journal_sha256=digest(journal),unfinished_committed_journal_entries=1,
        quality_acceptance=False,detector_calibrated=False,launch_ready=False))
    return parent,guard,bundle,checkpoint,tmp_path/'continuation'


def test_seal_preserves_original_and_does_not_retry(stopped):
    parent,guard,bundle,checkpoint,out=stopped;before=inventory(parent)
    spec=prepare(*stopped)
    assert spec['preserved_completed']==3 and spec['planned_new_attempts']==124
    assert spec['first_unstarted_index']==4 and not spec['new_model_load_authorized']
    assert inventory(parent)==before
    rows=read(out/'private/runs.json');failed=rows[-1]
    assert rows[:-1]==read(parent/'private/runs.json')
    assert failed['error_type']=='InfrastructureInterrupted' and failed['completion']=='incomplete'
    assert failed['committed_token_ids']==[0] and failed['text'] is None
    assert 'audit' not in failed and all('unavailable' in x for x in failed['raw_counts'])
    assert not (parent/'private'/failed['review_id']/'result.json').exists()
    assert audit_lineage(out,*stopped[:4])['new_attempts']==0
    with pytest.raises(FileExistsError):prepare(*stopped)
    with pytest.raises(ValueError,match='All 128'):finalize(out,*stopped[:4])


@pytest.mark.parametrize('change',['guard','checkpoint','order','token','header','partial_order','extra_folder','helper'])
def test_changed_parent_rejected_before_copy(stopped,change):
    parent,guard,bundle,checkpoint,out=stopped
    rows=read(parent/'private/runs.json');anchor=read(checkpoint)
    orphan=next(p for p in (parent/'private').iterdir() if p.is_dir() and p.name not in {r['review_id'] for r in rows})
    if change=='guard':
        d=read(guard/'result.json');d['cleanup_verified']=False;save(guard/'result.json',d)
    if change=='checkpoint':anchor['unstarted_attempts']=123;save(checkpoint,anchor)
    if change=='order':rows.reverse();save(parent/'private/runs.json',rows)
    if change=='token':
        p=parent/'private'/rows[0]['review_id']/'result.json';d=read(p);d['committed_token_ids']=[1,2];save(p,d)
    if change in ('header','partial_order'):
        p=orphan/'journal.jsonl';events=[json.loads(l) for l in p.read_text().splitlines()]
        if change=='header':events[0]['temperature']=1
        else:events[-1]['index']=4
        p.write_text(''.join(json.dumps(e)+'\n' for e in events));anchor['unfinished_journal_sha256']=digest(p);save(checkpoint,anchor)
    if change=='extra_folder':(parent/'private'/'dddddddddddd').mkdir()
    if change=='helper':
        p=parent/'public/plan.json';d=read(p);d['scripts_sha256']['balanced_inference.py']='0'*64;save(p,d)
        anchor['plan_sha256']=digest(p);save(checkpoint,anchor)
    with pytest.raises(ValueError):prepare(*stopped)
    assert not out.exists()


@pytest.mark.parametrize('change',['drop','seal','journal','plan','binding','prompt'])
def test_continuation_cannot_drop_failure_or_alter_preserved_data(stopped,change):
    parent,guard,bundle,checkpoint,out=stopped;prepare(*stopped,execute=True)
    rows=read(out/'private/runs.json')
    if change=='drop':save(out/'private/runs.json',rows[:-1])
    if change=='seal':
        p=out/'private'/rows[-1]['review_id']/'result.json';d=read(p);d['completion']='eos';save(p,d)
    if change=='journal':(out/'private'/rows[0]['review_id']/'journal.jsonl').write_text('changed')
    if change=='plan':
        p=out/'public/plan.json';d=read(p);d['schedule'].reverse();save(p,d)
    if change=='binding':
        p=out/'public/continuation.json';d=read(p);d['helpers_sha256']={};save(p,d)
        p=out/'public/plan.json';d=read(p);d['continuation_sha256']=digest(out/'public/continuation.json');save(p,d)
    if change=='prompt':save(out/'private/prompt-ids.json',{})
    with pytest.raises(ValueError):audit_lineage(out,*stopped[:4])


def append_synthetic_remainder(out,template):
    rows=read(out/'private/runs.json');allrows=read(template/'study/private/runs.json')
    for r in allrows[4:]:shutil.copytree(template/'study/private'/r['review_id'],out/'private'/r['review_id'])
    save(out/'private/runs.json',rows+allrows[4:])


def test_complete_fixture_keeps_one_failure_in_all_128_denominators(stopped,template):
    parent,guard,bundle,checkpoint,out=stopped;prepare(*stopped,execute=True)
    append_synthetic_remainder(out,template)
    result=finalize(out,*stopped[:4]);checked=audit(out,bundle)
    assert result['attempts']==128 and result['eos']==127 and result['errors']==1
    assert result['committed_tokens']==255 and checked['audited_runs']==127
    assert result['continuation']['new_attempts']==124 and not result['quality_acceptance']
    review=read(out/'private/blind-review.json');failures=[r for r in review if r['error_type']]
    assert len(failures)==1 and failures[0]['text'] is None
    assert all(not {'condition','key_slot','raw_counts'}&set(r) for r in review)


def test_plan_only_cannot_finalize_even_with_complete_fixture(stopped,template):
    out=stopped[-1];prepare(*stopped);append_synthetic_remainder(out,template)
    with pytest.raises(ValueError,match='Plan-only'):finalize(out,*stopped[:4])
