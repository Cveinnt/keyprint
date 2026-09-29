import json,re
import pytest
from build_balanced_review import render


def records():
    languages=['English','Spanish','French','Chinese']
    return [dict(review_id=str(i),case=dict(id=str(i),instruction='Task',source='</script><script>bad()</script>',language=languages[i%4]),
        rubric=dict(id=str(i),essential_facts=['fact']*6,qualifiers='No additions',format='Prose',expected_language=languages[i%4]),
        text='</script> & output',completion='cap',error_type=None,decode_error=None) for i in range(128)]


def test_multilingual_blank_review_and_safe_payload():
    source,page=render(records())
    payload=json.loads(re.search(r'id="review-data">(.*?)</script>',page,re.S)[1])
    assert {r['case']['language'] for r in payload['rows']}=={'English','Spanish','French','Chinese'}
    assert len(payload['rows'])==128 and 'ratings' not in payload
    assert payload['rows'][0]['text']=='</script> & output'
    assert '</script><script>bad()' not in page and 'Dolly' not in page
    assert "row.completion==='cap'" in page and 'keyprint-balanced-source-review:' in page
    assert all(not {'condition','key_slot','raw_counts'}&set(r) for r in json.loads(source))


def test_wrong_language_and_exposed_condition_rejected():
    rows=records();rows[0]['case']['language']='other'
    with pytest.raises(ValueError):render(rows)


def test_partial_view_requires_explicit_scope_and_visible_warning():
    rows=records()[:63]
    with pytest.raises(ValueError):render(rows)
    _,page=render(rows,allow_partial=True)
    assert 'INCOMPLETE STUDY: only 63 completed outputs shown out of 128 scheduled attempts.' in page
    assert 'All 128 attempts retained.' not in page
    rows=records();rows[0]['condition']='marked'
    with pytest.raises(ValueError):render(rows)
