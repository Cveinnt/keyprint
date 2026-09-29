import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from audit_paced_study import audit, digest
from build_paced_review import prepare
from validate_source_grounded import schedule


def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def fixture(tmp_path, failure=None):
    root=tmp_path/'candidate';original=tmp_path/'original'
    for path in (root/'public',root/'private',original/'public',original/'private'):path.mkdir(parents=True)
    cases=[{'id':str(i),'source':'Café </script><script>alert(1)</script>',
            'instruction':'Summarize the source.','prompt':'Source: café'} for i in range(16)]
    rubrics=[{'id':str(i),'essential_facts':['The text names a café.'],'format':'One sentence',
              'qualifiers':['No extra facts.']} for i in range(16)]
    for name,data in [('cases.json',cases),('rubrics.json',rubrics)]:
        save(original/'private'/name,data)
        # Candidate copies can use escaped Unicode without changing source text.
        (root/'private'/name).write_text(json.dumps(data,ensure_ascii=True,indent=2)+'\n')
    save(original/'public/plan.json',{'rubrics_commitment':{n:digest(original/'private'/n) for n in ('cases.json','rubrics.json')}})
    planned=schedule(cases)
    plan={'stage':'study','model_load_authorized_in_this_invocation':True,'schedule':planned,
          'settings':{'max_tokens':768},'original_plan_sha256':digest(original/'public/plan.json')}
    save(root/'public/plan.json',plan);save(root/'public/identity.json',{'profile_sha256':'profile'})
    rows=[];review=[]
    for i,item in enumerate(planned):
        uid=f'{i:012x}';folder=root/'private'/uid;folder.mkdir()
        (folder/'journal.jsonl').write_text('synthetic hash reconciliation fixture\n')
        result={'profile_sha256':'profile','condition':item['condition'],'completion':'eos',
                'committed_token_ids':[0],'text':'Café.','journal_sha256':digest(folder/'journal.jsonl')}
        if i==0 and failure=='forward':result.update(completion='incomplete',error_type='RuntimeError')
        if i==0 and failure=='decode':result.update(text=None,decode_error={'type':'UnicodeError','message':'fixture decode'})
        save(folder/'result.json',result)
        row=dict(item,review_id=uid,**{k:v for k,v in result.items() if k not in item},
                 result_sha256=digest(folder/'result.json'),raw_counts=[{'events':1,'trials':30,'ones':15,
                 'detector_calibrated':False,'token_path_only':True} for _ in range(2)])
        if not(i==0 and failure):row['audit']={'verified_draws':1,'native_heads_replayed':False,'quality_acceptance':False}
        rows.append(row)
        review.append({'review_id':uid,'case':cases[int(item['case'])],'rubric':rubrics[int(item['case'])],
                       **{k:row.get(k) for k in ('text','completion','error_type','decode_error')}})
    save(root/'private/runs.json',rows);save(root/'private/blind-review.json',list(reversed(review)))
    final={'attempts':128,'successful_runs':128-int(bool(failure)),'audited_runs':128-int(bool(failure)),
           'eos':128-int(failure=='forward'),'caps':0,'quality_acceptance':False,'detector_calibrated':False,'launch_ready':False}
    save(root/'public/results.json',final);refresh(root)
    return root,original


def refresh(root):
    p=root/'public/results.json';final=json.loads(p.read_text())
    for k,f in [('plan_sha256','public/plan.json'),('runs_sha256','private/runs.json'),('identity_sha256','public/identity.json')]:
        final[k]=digest(root/f)
    save(p,final)


def test_complete_cohort_reconciles_every_attempt_without_quality_claim(tmp_path):
    root,original=fixture(tmp_path);result=audit(root,original)
    assert result['attempts']==result['audited_runs']==result['committed_tokens']==128
    assert not result['native_heads_replayed'] and not result['quality_acceptance']
    assert len(result['outputs'])==128


@pytest.mark.parametrize('failure',['forward','decode'])
def test_failures_remain_in_complete_cohort_and_private_review(tmp_path,failure):
    root,original=fixture(tmp_path,failure);result=audit(root,original)
    assert result['successful_runs']==127 and result['audited_runs']==127
    assert result['errors']+result['decode_errors']==1
    out=tmp_path/'review';prepared=prepare(root,out,original)
    adapted=json.loads((out/'review-input.json').read_text());failed=next(r for r in adapted if r['review_id']=='000000000000')
    assert failed['error_type'] and prepared['rows']==128
    assert len(adapted)==128 and not prepared['quality_acceptance']


def test_review_keeps_original_order_blank_answers_and_escaped_source(tmp_path):
    root,original=fixture(tmp_path);out=tmp_path/'review'
    result=prepare(root,out,original);page=(out/'index.html').read_text()
    assert 'paced candidate source review' in page and 'Blank answers remain unreviewed' in page
    assert '</script><script>alert(1)</script>' not in page
    assert '\\u003c/script\\u003e' in page
    old=json.loads((root/'private/blind-review.json').read_text());new=json.loads((out/'review-input.json').read_text())
    assert [r['review_id'] for r in old]==[r['review_id'] for r in new]
    assert all('condition' not in r and 'key_slot' not in r and 'raw_counts' not in r for r in new)
    assert result['ratings']=='blank; no human acceptance inferred'
    with pytest.raises(FileExistsError):prepare(root,out,original)


@pytest.mark.parametrize('tamper',['partial','assignment','duplicate','raw','audit','summary','review_leak','review_drop','source','journal'])
def test_incomplete_or_inconsistent_cohorts_rejected(tmp_path,tamper):
    root,original=fixture(tmp_path)
    if tamper in ('partial','assignment','duplicate','raw','audit'):
        p=root/'private/runs.json';rows=json.loads(p.read_text())
        if tamper=='partial':rows.pop()
        if tamper=='assignment':rows[0]['condition']='marked'
        if tamper=='duplicate':rows[0]['review_id']=rows[1]['review_id']
        if tamper=='raw':rows[0]['raw_counts'][0]['ones']=31
        if tamper=='audit':rows[0]['audit']['verified_draws']=2
        save(p,rows)
    if tamper=='summary':
        p=root/'public/results.json';r=json.loads(p.read_text());r['audited_runs']=129;save(p,r)
    if tamper.startswith('review_'):
        p=root/'private/blind-review.json';rows=json.loads(p.read_text())
        if tamper=='review_leak':rows[0]['condition']='ordinary'
        else:rows.pop()
        save(p,rows)
    if tamper=='source':
        p=root/'private/cases.json';rows=json.loads(p.read_text());rows[0]['source']='Changed source';save(p,rows)
    if tamper=='journal':(root/'private/000000000000/journal.jsonl').write_text('changed')
    refresh(root)
    with pytest.raises(ValueError):audit(root,original)


def test_progress_alone_never_becomes_completed_evidence(tmp_path):
    root,original=fixture(tmp_path);(root/'public/results.json').unlink()
    with pytest.raises(FileNotFoundError):audit(root,original)
