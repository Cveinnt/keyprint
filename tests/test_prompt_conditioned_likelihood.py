import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    tools = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("prompt_conditioned_test", tools / "prompt_conditioned_likelihood.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_only_original_prompt_reaches_backend_and_provenance_is_explicit(module, monkeypatch):
    calls = []
    model, binding = object(), object()
    backend = SimpleNamespace(model=model, encode_prompt=lambda prompt: calls.append(prompt) or [4, 5])
    def replay(adapted, actual_binding, text):
        assert adapted.model is model and actual_binding is binding and text == "Visible text"
        prefix = adapted.encode_prompt(module.PREFIX)
        return {"fixed_prefix_ids": prefix, "original_prompt_used": False,
                "key_used_for_model_inference": False, "token_ids": [8], "heads": ["unchanged"]}
    monkeypatch.setattr(module, "surrogate_measure", replay)
    result = module.measure(backend, binding, "Visible text", "Explain a seed. 🌱")
    assert calls == ["Explain a seed. 🌱"]
    assert result["conditioning_prefix_ids"] == [4, 5]
    assert "fixed_prefix_ids" not in result
    assert result["original_prompt_used"] and not result["private_generation_data_used"]
    assert not result["key_used_for_model_inference"]
    assert result["heads"] == ["unchanged"]
    assert result["original_prompt_sha256"] == hashlib.sha256("Explain a seed. 🌱".encode()).hexdigest()


@pytest.mark.parametrize("prompt", [None, b"bytes", "", "   ", "x" * 16001])
def test_invalid_prompt_rejected_before_model_use(module, prompt):
    with pytest.raises(ValueError, match="Original prompt"):
        module.PromptConditionedModel(None, prompt)


def test_changed_frozen_prefix_fails_closed(module):
    adapter = module.PromptConditionedModel(SimpleNamespace(model=None), "Original prompt")
    with pytest.raises(ValueError, match="prefix differs"):
        adapter.encode_prompt("Unexpected new conditioning contract")
