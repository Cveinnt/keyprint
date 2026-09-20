import hashlib
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0,str(TOOLS))
try:
    import serving_confirmation as confirmation
finally:
    sys.path.remove(str(TOOLS))


def corpus(tmp_path,monkeypatch,*,categories=8):
    rows=[{'category':f'category-{c}','instruction':f'Task {c}-{i}', 'context':f'Context {i}', 'response':'not used'}
          for c in range(categories) for i in range(3)]
    rows.append({'category':'category-0','instruction':'x'*8001,'context':'','response':'not used'})
    path=tmp_path/'corpus.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    monkeypatch.setattr(confirmation,'CORPUS_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    return path,rows


def test_selection_is_deterministic_category_balanced_and_preserves_task(tmp_path,monkeypatch):
    path,rows=corpus(tmp_path,monkeypatch)
    first=confirmation.select_cases(path)
    assert first==confirmation.select_cases(path)
    assert len(first)==8 and len({c['category'] for c in first})==8
    for case in first:
        row=rows[case['source_index']]
        assert case['prompt']==row['instruction']+'\n\nContext:\n'+row['context']
        assert len(case['prompt'])<=8000 and 'response' not in case
        assert case['max_tokens']==192


def test_changed_corpus_and_missing_category_cannot_silently_substitute_workload(tmp_path,monkeypatch):
    path,_=corpus(tmp_path,monkeypatch,categories=7)
    with pytest.raises(ValueError,match='eight'):confirmation.select_cases(path)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='hash'):confirmation.select_cases(path)
