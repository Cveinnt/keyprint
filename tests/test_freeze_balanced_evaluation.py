import json
from pathlib import Path
import pytest
from freeze_balanced_evaluation import material, freeze, validate, normalized

SOURCE=Path(__file__).parents[1]/'research/balanced-evaluation/cases.json'


def test_all_cases_preserve_six_source_facts_and_language():
    records=json.loads(SOURCE.read_text());cases,rubrics=material(records)
    assert len(cases)==16 and sum(len(r['essential_facts']) for r in rubrics)==96
    for c,r,raw in zip(cases,rubrics,records):
        assert c['prompt'].endswith(c['source'])
        assert c['source'].splitlines()==r['essential_facts']==raw['facts']
        assert c['language']==r['expected_language']


@pytest.mark.parametrize('change',['duplicate','language','missing_fact','missing_format'])
def test_bad_material_rejected(change):
    records=json.loads(SOURCE.read_text())
    if change=='duplicate':records[1]=records[0]
    if change=='language':records[0]['language']='German'
    if change=='missing_fact':records[0]['facts'].pop()
    if change=='missing_format':records[0]['format']=''
    with pytest.raises(ValueError):material(records)


def bundle(tmp_path):
    prior=tmp_path/'prior.json';prior.write_text('[]')
    root=tmp_path/'bundle';freeze(SOURCE,[prior],root)
    return root


def test_complete_schedule_and_no_overwrite(tmp_path):
    root=bundle(tmp_path);m=validate(root)
    assert len(m['schedule'])==128 and m['total_fact_checks_planned']==768
    assert len(m['decoy_key_sha256'])==199 and not m['generation_executed']
    signatures={(r['case'],r['key_slot'],r['condition']) for r in m['schedule']}
    assert len(signatures)==128
    with pytest.raises(FileExistsError):freeze(SOURCE,[tmp_path/'prior.json'],root)


@pytest.mark.parametrize('path',['private/cases.json','private/rubrics.json','private/decoys.json','private/key-0','public/protocol.json'])
def test_tampering_rejected(tmp_path,path):
    root=bundle(tmp_path);p=root/path;p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):validate(root)


def test_exact_normalized_prior_overlap_rejected_before_key_creation(tmp_path):
    records=json.loads(SOURCE.read_text());cases,_=material(records)
    prior=tmp_path/'prior.json';prior.write_text(json.dumps([{'source':'  '+cases[0]['source'].upper()+'  '}]))
    with pytest.raises(ValueError,match='overlaps'):freeze(SOURCE,[prior],tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()
    assert normalized(' e\u0301  x ')==normalized('É x')


@pytest.mark.parametrize('change',['claim','missing_hash','count','schedule'])
def test_manifest_cannot_upgrade_or_drop_required_evidence(tmp_path,change):
    root=bundle(tmp_path);p=root/'public/manifest.json';m=json.loads(p.read_text())
    if change=='claim':m['quality_acceptance']=True
    if change=='missing_hash':del m['files_sha256']['private/rubrics.json']
    if change=='count':m['total_fact_checks_planned']=1
    if change=='schedule':m['schedule'].pop()
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):validate(root)
