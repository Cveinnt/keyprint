import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def recovery(monkeypatch):
    directory=Path(__file__).parents[1]/'tools';monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('recovery_test',directory/'recover_prompt_null.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def parent(tmp_path,recovery,missing_error=None):
    public=tmp_path/'public';public.mkdir();tasks=[{'source_index':i} for i in range(500)]
    rows=[]
    for i in range(500):
        row={'id':f'null-{i:03d}','source_index':i}
        if i==499:row['error']=missing_error or recovery.DISK_ERROR
        else:
            row['flagged']=False
            for suffix,field in (('.heads.json','heads_sha256'),('.scores.json','scores_sha256')):
                file=tmp_path/(row['id']+suffix);file.write_text('{}');row[field]=recovery.sha(file.read_bytes())
        rows.append(row)
    (public/'plan.json').write_text(json.dumps({'tasks':tasks}))
    (public/'summary.json').write_text(json.dumps({'status':'incomplete','fatal':None,'available':499,'unavailable':1}))
    (public/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    audit={'status':'pass','plan_sha256':recovery.sha((public/'plan.json').read_bytes()),
           'summary_sha256':recovery.sha((public/'summary.json').read_bytes()),
           'results_prefix_sha256':recovery.sha((public/'results.jsonl').read_bytes()),
           'results_prefix_bytes':len((public/'results.jsonl').read_bytes())}
    (public/'integrity.json').write_text(json.dumps(audit))
    return rows


def test_recovers_all_and_only_unmeasured_disk_failures(recovery,tmp_path):
    rows=parent(tmp_path,recovery)
    _,actual,missing=recovery.parent_state(tmp_path)
    assert actual==rows and missing==[499]


@pytest.mark.parametrize('damage',['model_error','partial_heads','partial_scores','changed_log','changed_heads','changed_summary'])
def test_other_failures_or_changed_evidence_cannot_be_recovered(recovery,tmp_path,damage):
    parent(tmp_path,recovery,{'type':'ValueError','message':'model failed'} if damage=='model_error' else None)
    if damage=='partial_heads':(tmp_path/'null-499.heads.json').write_text('{}')
    elif damage=='partial_scores':(tmp_path/'null-499.scores.json').write_text('[]')
    elif damage=='changed_log':(tmp_path/'public/results.jsonl').write_text('{}\n')
    elif damage=='changed_heads':(tmp_path/'null-000.heads.json').write_text('[]')
    elif damage=='changed_summary':(tmp_path/'public/summary.json').write_text('{"status":"completed"}')
    with pytest.raises(ValueError):recovery.parent_state(tmp_path)


def test_compact_receipts_round_trip_without_overwriting(recovery,tmp_path):
    path=tmp_path/'scores.json';value={'values':[1e-300,-1.25,0.0],'text':'你好'}
    recovery.compact(path,value)
    assert json.loads(path.read_text())==value
    with pytest.raises(FileExistsError):recovery.compact(path,{'replacement':True})


def test_recovery_audit_rejects_changed_retained_records_and_lost_failure(recovery):
    from audit_prompt_null_recovery import check_record
    original={'id':'null-000','source_index':4,'flagged':False,'working_log_ratios':[-1.,-2.]}
    check_record(original,{**original,'execution_origin':'retained'})
    with pytest.raises(ValueError,match='Retained'):
        check_record(original,{**original,'execution_origin':'retained','flagged':True})
    missing={'id':'null-001','source_index':8,'error':recovery.DISK_ERROR}
    valid={'id':'null-001','source_index':8,'execution_origin':'recovery','original_error':recovery.DISK_ERROR}
    check_record(missing,valid)
    for field,value in (('source_index',9),('execution_origin','retained'),('original_error',{})):
        with pytest.raises(ValueError,match='origin'):
            check_record(missing,{**valid,field:value})


def test_preflight_selects_extreme_and_median_completed_sizes_only(recovery,tmp_path):
    from preflight_prompt_null_memory import select_completed
    rows=[]
    for index,size in enumerate([20,40,30,10]):
        path=tmp_path/f'case-{index}.heads.json'
        path.write_text(json.dumps({'conditioning_prefix_ids':[1]*size,'token_ids':[2]*5}))
        rows.append({'id':f'case-{index}','heads_sha256':recovery.sha(path.read_bytes())})
    rows.append({'id':'never-read','error':recovery.DISK_ERROR})
    assert [r['id'] for r in select_completed(tmp_path,rows)]==['case-3','case-2','case-1']
    (tmp_path/'case-0.heads.json').write_text('{}')
    with pytest.raises(ValueError,match='heads differ'):select_completed(tmp_path,rows)
