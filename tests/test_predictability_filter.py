import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("predictability_filter", Path(__file__).parents[1]/"tools/predictability_filter.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_predictive_heads_exclude_the_token_being_scored_and_future_tokens():
    assert module.causal_inputs([8, 9], [1, 2, 3]) == ([8], [9, 1, 2])
    assert module.causal_inputs([8], [1]) == ([], [8])
    for prefix, text in (([], [1]), ([8], [])):
        with pytest.raises(ValueError, match="Nonempty"):
            module.causal_inputs(prefix, text)
    assert module.fixed_batch([8,9]) == [8,9]+[0]*62
    assert module.fixed_batch(list(range(64))) == list(range(64))
    with pytest.raises(ValueError, match="bounded"):
        module.fixed_batch([])


def test_entropy_on_point_mass_and_fair_choice_excludes_model_padding():
    raw = np.full((1, 151936), -np.inf, dtype=np.float32)
    raw[0, 1] = 0
    raw[0, 151800] = 100  # Padding is not an admitted choice.
    assert module.entropy_bits(raw) == 0
    raw[0, 2] = 0
    assert module.entropy_bits(raw) == pytest.approx(1)


def test_selection_excludes_repeat_contexts_and_uses_inclusive_frozen_cutoff():
    profile = SimpleNamespace(classes=(None,b'a',b'b',b'c'), config=SimpleNamespace(history=1))
    assert module.eligible_positions(profile, [0,1,2,1,2,3]) == [1,2,3]
    assert module.selected_positions([1,2,3], [0,.49999,.5,1.,2.,3.]) == [2,3]
    with pytest.raises(ValueError, match="nonnegative"):
        module.selected_positions([0], [float("nan")])
    with pytest.raises(ValueError, match="Distinct"):
        module.selected_positions([0,0], [1.])


def test_selected_bits_preserve_token_event_alignment_and_key_blind_selection():
    profile = SimpleNamespace(classes=(None,b'a',b'b'), config=SimpleNamespace(history=1,layers=2))
    text = "ab"
    class Binding:
        def replay_text(self, text, key):
            return SimpleNamespace(token_ids=[0,1,2,1,2], events=[
                SimpleNamespace(bits=[key,1]), SimpleNamespace(bits=[1,key]),
                SimpleNamespace(bits=[key,key]), SimpleNamespace(bits=None)])
    binding = Binding()
    binding.profile = profile
    selection = {"text_sha256":hashlib.sha256(text.encode()).hexdigest(),
                 "token_ids":[0,1,2,1,2], "eligible_positions":[1,2,3],
                 "entropy_bits":[0,.1,1.,.5,2.], "selected_positions":[2,3]}
    assert module.selected_bits(binding,text,0,selection).tolist() == [[1,0],[0,0]]
    assert module.selected_bits(binding,text,1,selection).tolist() == [[1,1],[1,1]]
    with pytest.raises(ValueError, match="another text"):
        module.selected_bits(binding,"changed",0,selection)
    with pytest.raises(ValueError, match="frozen entropy"):
        module.selected_bits(binding,text,0,{**selection,"selected_positions":[1,2,3]})


@pytest.mark.parametrize("passed,dependencies,message", [
    (False, {}, "power-screen gate did not pass"),
    (True, {"predictability_filter.py":"changed"}, "Frozen candidate source differs"),
    (True, {}, "Candidate integrity and prefix stability must pass"),
])
def test_null_inference_cannot_start_without_passed_frozen_candidate(tmp_path, passed, dependencies, message):
    development = tmp_path/"development/public"
    development.mkdir(parents=True)
    (development/"plan.json").write_text(json.dumps({"dependencies_sha256":dependencies}))
    (development/"summary.json").write_text(json.dumps({"status":"completed", "expand_to_null_controls":passed}))
    (development/"integrity.json").write_text(json.dumps({"status":"failed", "candidate_hashes":dependencies}))
    output = tmp_path/"must-not-exist"
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1]/"tools/develop_predictability_null.py"),
        "--development",str(development.parent), "--source",str(tmp_path/"no-source"),
        "--null-study",str(tmp_path/"no-study"), "--model",str(tmp_path/"no-model"),
        "--output",str(output)], capture_output=True,text=True)
    assert result.returncode != 0 and message in result.stderr
    assert not output.exists()
