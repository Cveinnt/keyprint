import json
from pathlib import Path
from threading import Event
import pytest
from keyprint import Keyprint, Generation, Inspection
from keyprint.comparison import compare
from keyprint.cancellation import _CancellationRequested


class Model:
    def __init__(self): self.calls = []
    def generate(self, prompt, *, condition, output, **kwargs):
        self.calls.append((prompt, condition))
        Path(output).mkdir()
        return Generation("Unchanged 🌱" if condition == "ordinary" else "Grown 🌱",
            {"completion": "eos", "usage": {"completion_tokens": 4, "private": "SECRET"}}, output)
    def inspect(self, text, *, key=None):
        return Inspection(1, 8 if key is None else 7, 16,
                          {"kind": "literal_diagnostic", "private": "SECRET"})


def test_real_pair_contract_shared_by_ui_and_export(tmp_path):
    model = Model()
    pair = compare(model, "Seed 🌱", output=tmp_path/'private')
    assert model.calls == [("Seed 🌱", "ordinary"), ("Seed 🌱", "marked")]
    assert pair.ordinary == "Unchanged 🌱" and pair.marked == "Grown 🌱"
    data = pair.to_dict()
    data['experiment']['outputs']['marked']['text'] = 'modified copy'
    assert pair.marked == "Grown 🌱"
    entry = pair.export(tmp_path/'site')
    exported = json.loads((entry.parent/'replay.json').read_text())
    assert exported['experiment']['outputs']['marked']['text'] == pair.marked
    assert 'SECRET' not in json.dumps(exported)
    assert not list(entry.parent.glob('*.key'))
    assert 'data-mode="replay"' in entry.read_text()
    assert 'src="./app.js"' in entry.read_text()
    assert (entry.parent/'app.js').read_bytes() == Path('src/keyprint/web/app.js').read_bytes()
    assert len((tmp_path/'private'/'comparison-control.key').read_bytes()) == 32
    with pytest.raises(FileExistsError): pair.export(entry.parent)


@pytest.mark.parametrize('prompt,cap', [('',32),('  ',32),('a'*16001,32),('a',True),('a',0),('a',1025)])
def test_invalid_requests_do_no_inference_or_artifact_work(tmp_path,prompt,cap):
    model=Model()
    with pytest.raises(ValueError): compare(model,prompt,max_tokens=cap,output=tmp_path/'run')
    assert not model.calls and not (tmp_path/'run').exists()


def test_cancel_before_start_does_no_work(tmp_path):
    event=Event(); event.set(); model=Model()
    with pytest.raises(_CancellationRequested):
        compare(model,'Seed',output=tmp_path/'run',cancel_event=event)
    assert not model.calls and not (tmp_path/'run').exists()


def test_keyprint_method_uses_identical_contract(monkeypatch,tmp_path):
    wm=object.__new__(Keyprint); wm._closed=False
    model=Model()
    monkeypatch.setattr(wm,'generate',model.generate)
    monkeypatch.setattr(wm,'inspect',model.inspect)
    pair=wm.compare('Seed',output=tmp_path/'run')
    assert pair.marked == 'Grown 🌱'
    wm._closed=True
    with pytest.raises(RuntimeError, match='closed'): wm.compare('Seed',output=tmp_path/'closed')


def test_world_readable_output_rejected(tmp_path):
    run=tmp_path/'run'; run.mkdir(mode=0o755); model=Model()
    with pytest.raises(ValueError,match='private'): compare(model,'Seed',output=run)
    assert not model.calls and not list(run.iterdir())
