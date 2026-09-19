import hashlib
import importlib.util
import json
from pathlib import Path
import math
import pytest

spec = importlib.util.spec_from_file_location("capacity_analysis", Path(__file__).parents[1] / "tools/analyze_mlx_capacity.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_two_choice_channel_and_point_mass_information():
    result = analysis.metrics([.5,.5],[.25,.75],[-.5,.5],1)
    assert result["base_entropy_bits"] == 1
    assert result["expected_bit_lift"] == .25
    assert result["base_score_variance"] == .25
    assert result["total_variation"] == .25
    assert result["selected_log2_ratio"] == pytest.approx(math.log2(1.5))
    assert result["marked_kl_base_bits"] == pytest.approx(.25*math.log2(.5)+.75*math.log2(1.5))
    point = analysis.metrics([1.],[1.],[8],0)
    assert point["base_entropy_bits"] == point["expected_bit_lift"] == point["marked_kl_base_bits"] == 0
    assert point["observed_centered_bits"] == 8  # High raw bits can carry zero marking information.


def test_unnormalized_support_cannot_be_silently_repaired():
    with pytest.raises(ValueError, match="normalization"):
        analysis.metrics([.5,.4],[.25,.75],[-.5,.5],0)


def test_modified_or_truncated_source_journal_rejected(tmp_path):
    path=tmp_path/"journal.jsonl"
    first=json.dumps({"sequence":0,"previous_sha256":"0"*64,"event":{"kind":"first"}}).encode()+b"\n"
    second=json.dumps({"sequence":1,"previous_sha256":hashlib.sha256(first).hexdigest(),"event":{"kind":"second"}}).encode()+b"\n"
    path.write_bytes(first+second)
    assert len(analysis.read_journal(path))==2
    path.write_bytes(first.replace(b'first',b'other')+second)
    with pytest.raises(ValueError, match="chain"):
        analysis.read_journal(path)
    path.write_bytes(first+second.rstrip())
    with pytest.raises(ValueError, match="chain"):
        analysis.read_journal(path)


def test_weighted_source_resolution_checks_prompt_and_key_identity(tmp_path):
    prompt = "Keep the original question."
    key = b"a" * 32
    (tmp_path / "owner-1.key").write_bytes(key)
    task = {"source_index": 12, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    plan = {"tasks": [task], "key_commitments": ["unused", hashlib.sha256(key).hexdigest()]}
    row = {"weighted": [], "source_index": 12, "key_index": 1}
    assert analysis.source_prompt(plan, row) == prompt
    assert analysis.source_key(tmp_path, plan, row) == key
    with pytest.raises(ValueError, match="Exactly one"):
        analysis.source_prompt({**plan, "tasks": [task, task]}, row)
    with pytest.raises(ValueError, match="prompt hash"):
        analysis.source_prompt({**plan, "tasks": [{**task, "prompt": "different"}]}, row)
    (tmp_path / "owner-1.key").write_bytes(b"b" * 32)
    with pytest.raises(ValueError, match="key commitment"):
        analysis.source_key(tmp_path, plan, row)
    with pytest.raises(ValueError, match="key index"):
        analysis.source_key(tmp_path, plan, {**row, "key_index": True})
    assert analysis.source_prompt({"prompt_suffix": " suffix"}, {"prompt": "original"}) == "original suffix"
