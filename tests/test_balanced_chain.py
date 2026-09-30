"""Synthetic lineage and bounded-batch tests; no model or quality acceptance."""
from pathlib import Path
import json
import shutil
import pytest
from test_balanced_study import template
from test_balanced_continuation import stopped
from balanced_continuation import prepare, inventory
from balanced_chain import prepare_chain, verify_history, audit_link, finish_batch
from balanced_study import read, audit
from balanced_inference import save
from audit_paced_study import digest


def append_rows(output, template, start, stop):
    rows = read(output/'private/runs.json'); original = read(template/'study/private/runs.json')
    assert len(rows) == start
    for row in original[start:stop]:
        shutil.copytree(template/'study/private'/row['review_id'], output/'private'/row['review_id'])
    save(output/'private/runs.json', rows + original[start:stop])


def interrupt(output, template, index):
    row = read(template/'study/private/runs.json')[index]
    folder = output/'private'/row['review_id']; folder.mkdir()
    source = template/'study/private'/row['review_id']/'journal.jsonl'
    (folder/'journal.jsonl').write_text(''.join(source.read_text().splitlines(keepends=True)[:4]))
    return folder/'journal.jsonl'


def anchor(root, guard, checkpoint, journal=None):
    data = {key:digest(path) for key,path in [
        ('plan_sha256',root/'public/plan.json'), ('runs_sha256',root/'private/runs.json'),
        ('identity_sha256',root/'public/identity.json'), ('supervisor_sha256',guard/'result.json')]}
    if journal:data['unfinished_journal_sha256']=digest(journal)
    save(checkpoint,data)
    return dict(root=root, supervisor=guard, checkpoint=checkpoint)


@pytest.fixture
def history(stopped, template, tmp_path):
    parent, guard, bundle, checkpoint, first = stopped
    prepare(*stopped, execute=True); append_rows(first,template,4,7)
    journal=interrupt(first,template,7)
    nextguard=tmp_path/'guard1';nextguard.mkdir()
    save(nextguard/'result.json',dict(status='stopped',reason='system_pressure',exit_code=-15,cleanup_verified=True))
    stages=[dict(root=parent,supervisor=guard,checkpoint=checkpoint),
            anchor(first,nextguard,tmp_path/'checkpoint1.json',journal)]
    return stages,bundle,tmp_path/'chain'


def test_two_interruptions_preserved_and_originals_unchanged(history):
    stages,bundle,out=history;before=[inventory(s['root']) for s in stages]
    spec=prepare_chain(stages,bundle,out,max_new_attempts=4)
    rows=read(out/'private/runs.json')
    assert len(rows)==8 and len([r for r in rows if r.get('error_type')])==2
    assert rows[-1]['text'] is None and rows[-1]['committed_token_ids']==[0]
    assert 'audit' not in rows[-1] and all('unavailable' in s for s in rows[-1]['raw_counts'])
    assert spec['first_unstarted_index']==8 and spec['max_new_attempts']==4
    assert [inventory(s['root']) for s in stages]==before
    assert audit_link(out,stages,verify_history(stages,bundle))['new_attempts']==0
    with pytest.raises(FileExistsError):prepare_chain(stages,bundle,out,max_new_attempts=4)
    with pytest.raises(ValueError,match='planned new attempt'):finish_batch(out,stages,bundle)


def paused_stage(history,template,tmp_path):
    stages,bundle,out=history
    prepare_chain(stages,bundle,out,max_new_attempts=2,execute=True)
    append_rows(out,template,8,10);marker=finish_batch(out,stages,bundle)
    assert marker['next_index']==10
    guard=tmp_path/'pauseguard';guard.mkdir()
    save(guard/'result.json',dict(status='completed',exit_code=0,cleanup_verified=True))
    return stages+[anchor(out,guard,tmp_path/'pausecheckpoint.json')],bundle


def test_planned_pause_continues_at_boundary_without_fabricating_failure(history,template,tmp_path):
    stages,bundle=paused_stage(history,template,tmp_path);out=tmp_path/'next'
    spec=prepare_chain(stages,bundle,out,max_new_attempts=3,execute=True)
    assert spec['sealed_interrupted']==0 and spec['first_unstarted_index']==10
    assert not (out/'public/batch-complete.json').exists()
    assert digest(out/'public/parent-pause-003.json')==digest(stages[-1]['root']/'public/batch-complete.json')
    append_rows(out,template,10,13);finish_batch(out,stages,bundle)
    assert len([r for r in read(out/'private/runs.json') if r.get('error_type')])==2
    with pytest.raises(FileExistsError):finish_batch(out,stages,bundle)


@pytest.mark.parametrize('change',['old_text','new_journal','checkpoint','helper','cleanup','prompt','extra_attempt','false_negative'])
def test_corrupted_parent_rejected_before_copy(history,change):
    stages,bundle,out=history;root=stages[-1]['root'];rows=read(root/'private/runs.json')
    if change=='old_text':
        p=root/'private'/rows[0]['review_id']/'result.json';d=read(p);d['text']='changed';save(p,d)
    if change=='new_journal':(root/'private'/rows[-1]['review_id']/'journal.jsonl').write_text('changed')
    if change=='checkpoint':
        p=stages[-1]['checkpoint'];d=read(p);d['runs_sha256']='0'*64;save(p,d)
    if change=='helper':
        p=root/'public/continuation.json';d=read(p);d['helpers_sha256']={};save(p,d)
    if change=='cleanup':
        p=stages[-1]['supervisor']/'result.json';d=read(p);d['cleanup_verified']=False;save(p,d)
    if change=='prompt':save(root/'private/prompt-ids.json',{})
    if change=='extra_attempt':(root/'private'/'eeeeeeeeeeee').mkdir()
    if change=='false_negative':
        rows[3]['raw_counts']=[{'ones':0,'events':0,'trials':0}]*2;save(root/'private/runs.json',rows)
    with pytest.raises((ValueError,KeyError)):prepare_chain(stages,bundle,out,max_new_attempts=4)
    assert not out.exists()


@pytest.mark.parametrize('change',['drop_failure','failure_text','helpers','extra_new','plan_only_execution'])
def test_corrupted_child_rejected(history,template,change):
    stages,bundle,out=history
    prepare_chain(stages,bundle,out,max_new_attempts=1,execute=change!='plan_only_execution')
    rows=read(out/'private/runs.json')
    if change=='drop_failure':save(out/'private/runs.json',rows[:-1])
    if change=='failure_text':
        p=out/'private'/rows[-1]['review_id']/'result.json';d=read(p);d['text']='invented';save(p,d)
    if change=='helpers':
        p=out/'public/chain-link-002.json';d=read(p);d['helpers_sha256']={};save(p,d)
        p=out/'public/plan.json';d=read(p);d['chain_link_sha256']=digest(out/'public/chain-link-002.json');save(p,d)
    if change=='extra_new':append_rows(out,template,8,10)
    if change=='plan_only_execution':append_rows(out,template,8,9)
    with pytest.raises(ValueError):audit_link(out,stages,verify_history(stages,bundle))


def test_pause_receipt_cannot_be_rewritten(history,template,tmp_path):
    stages,bundle=paused_stage(history,template,tmp_path);out=tmp_path/'next'
    prepare_chain(stages,bundle,out,max_new_attempts=1,execute=True)
    (out/'public/parent-pause-003.json').write_text('{}')
    with pytest.raises(ValueError,match='pause receipt'):audit_link(out,stages,verify_history(stages,bundle))


def test_unindexed_attempt_prevents_finalization(history,template):
    stages,bundle,out=history;prepare_chain(stages,bundle,out,max_new_attempts=1,execute=True)
    append_rows(out,template,8,9);interrupt(out,template,9)
    with pytest.raises(ValueError,match='Unindexed'):finish_batch(out,stages,bundle)


def test_full_synthetic_cohort_retains_two_failures(history,template):
    stages,bundle,out=history;prepare_chain(stages,bundle,out,max_new_attempts=120,execute=True)
    append_rows(out,template,8,128);result=finish_batch(out,stages,bundle)
    assert result['attempts']==128 and result['errors']==2 and result['eos']==126
    assert audit(out,bundle)['audited_runs']==126
    review=read(out/'private/blind-review.json')
    assert len(review)==128 and sum(r['text'] is None for r in review)==2
    assert all(not {'condition','key_slot','raw_counts'}&set(r) for r in review)


def test_invalid_batch_and_ancestor_output_rejected(history):
    stages,bundle,out=history
    for n in (0,121,True,-1):
        with pytest.raises(ValueError,match='batch length'):prepare_chain(stages,bundle,out,max_new_attempts=n)
    with pytest.raises(ValueError,match='disjoint'):
        prepare_chain(stages,bundle,stages[-1]['root']/'child',max_new_attempts=1)


def test_final_interruption_can_finish_without_new_generation(history,template):
    stages,bundle,out=history;root=stages[-1]['root']
    orphan=next(p for p in (root/'private').iterdir() if p.is_dir() and not (p/'result.json').exists())
    shutil.rmtree(orphan);append_rows(root,template,7,127)
    journal=interrupt(root,template,127)
    stages[-1]=anchor(root,stages[-1]['supervisor'],stages[-1]['checkpoint'],journal)
    spec=prepare_chain(stages,bundle,out,max_new_attempts=0,execute=True)
    assert spec['first_unstarted_index']==128
    result=finish_batch(out,stages,bundle)
    assert result['attempts']==128 and result['errors']==2 and result['continuation']['new_attempts']==0


def test_stop_between_attempts_does_not_invent_failure(history):
    stages,bundle,out=history;root=stages[-1]['root']
    orphan=next(p for p in (root/'private').iterdir() if p.is_dir() and not (p/'result.json').exists())
    shutil.rmtree(orphan)
    stages[-1]=anchor(root,stages[-1]['supervisor'],stages[-1]['checkpoint'])
    spec=prepare_chain(stages,bundle,out,max_new_attempts=1)
    assert spec['first_unstarted_index']==7 and spec['sealed_interrupted']==0
    assert len([r for r in read(out/'private/runs.json') if r.get('error_type')])==1
