import pytest
from keyprint.trace import token_trace


def test_split_unicode_and_control_are_exact_not_replacement_characters():
    pieces=[b'A ',b'\xe4',b'\xb8',b'\xad',None]
    report={'committed_token_ids':[0,1,2,3,4],'completion':'eos'}
    steps=token_trace(report,'A 中',pieces)
    assert ''.join(s.text for s in steps)=='A 中'
    assert [s.text for s in steps]==['A ','','','中','']
    assert [s.kind for s in steps]==['text','pending_bytes','pending_bytes','text','control']
    assert (steps[3].start,steps[3].end)==(2,3)
    assert steps[-1].bytes_hex is None


def test_capped_suffix_is_retained_without_fabricating_text():
    r={'payload':{'committed_token_ids':[0,1],'completion':'length'}}
    steps=token_trace(r,'A',[b'A',b'\xe4'])
    assert steps[-1].bytes_hex=='e4' and steps[-1].text==''
    r['payload']['completion']='eos'
    with pytest.raises(ValueError,match='Incomplete'):token_trace(r,'A',[b'A',b'\xe4'])


@pytest.mark.parametrize('ids,text', [([0],'wrong'),([2],'A'),([True],'A')])
def test_invalid_or_inconsistent_trace_fails(ids,text):
    with pytest.raises(ValueError):token_trace({'committed_token_ids':ids,'completion':'eos'},text,[b'A'])


def test_public_trace_adds_no_backend_calls_and_validates_before_inference(tmp_path):
    from types import SimpleNamespace
    from keyprint import Keyprint, Generation
    calls=[]
    original=Generation('A',{'committed_token_ids':[0,1],'completion':'eos'},tmp_path)
    def generate(prompt, **settings):
        calls.append((prompt,settings))
        return original
    wm=object.__new__(Keyprint)
    wm._closed=False; wm._key=bytes(32)
    wm._backend=SimpleNamespace(generate=generate,binding=SimpleNamespace(pieces=[b'A',None]))
    plain=wm.generate('Prompt',max_tokens=2)
    traced=wm.generate('Prompt',max_tokens=2,trace=True)
    assert len(calls)==2 and calls[0]==calls[1]
    assert plain is original and plain.trace is None
    assert traced.text==plain.text and traced.report is plain.report
    assert ''.join(s.text for s in traced.trace)==plain.text
    with pytest.raises(TypeError,match='boolean'):wm.generate('Prompt',trace='yes')
    assert len(calls)==2
