import json
from pathlib import Path

import pytest

from keyprint import Generation, Keyprint
from keyprint.rewrite import anthropic_text, fidelity_checks, openai_text, plain_text


def chat(text="The meeting is at 10:30.", **message):
    return {"id": "test", "created": 0, "model": "fixture", "object": "chat.completion", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text, **message}}]}


def claude(**extra):
    return {"id": "test", "model": "fixture", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "The meeting is at 10:30."}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 4, "output_tokens": 9}, **extra}


def test_actual_provider_sdk_objects_are_accepted_without_network():
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    from openai.types.chat import ChatCompletion
    from anthropic.types import Message
    assert openai_text(ChatCompletion.model_validate(chat())) == "The meeting is at 10:30."
    assert anthropic_text(Message.model_validate(claude())) == "The meeting is at 10:30."


@pytest.mark.parametrize("field", ["tool_calls", "function_call", "refusal", "annotations", "audio", "reasoning_content"])
def test_chat_nontext_context_rejected(field):
    with pytest.raises(ValueError):
        openai_text(chat(**{field: ["unsupported"]}))


@pytest.mark.parametrize("kind", ["thinking", "tool_use", "redacted_thinking"])
def test_anthropic_nontext_rejected(kind):
    with pytest.raises(ValueError):
        anthropic_text(claude(content=[{"type": kind}]))


def test_truncated_provider_response_rejected_before_generation():
    response = chat()
    response["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        openai_text(response)
    with pytest.raises(ValueError):
        anthropic_text(claude(stop_reason="max_tokens"))


def test_responses_api_text_and_refusal():
    response = {"object": "response", "status": "completed", "output": [
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Hello", "annotations": []}]}]}
    assert openai_text(response) == "Hello"
    response["output"][0]["content"][0]["type"] = "refusal"
    with pytest.raises(ValueError):
        openai_text(response)


@pytest.mark.parametrize("text", ["", "{}", "[1]", "```python\nprint(1)\n```"])
def test_unsupported_input(text):
    with pytest.raises(ValueError):
        plain_text(text)


def test_checks_do_not_pretend_to_measure_meaning():
    checks = fidelity_checks("Do not pay 12 dollars.", "Pay 12 dollars.", completion="eos")
    assert checks["numbers_preserved"] is True
    assert checks["meaning_preservation"] == "not_measured"
    assert fidelity_checks("12 dollars", "13 dollars", completion="length")["numbers_preserved"] is False


def test_original_and_candidate_retained(tmp_path):
    kp = Keyprint(key=bytes(range(32)))
    def generate(prompt, *, max_tokens, output, condition):
        assert '"Meet at 10:30."' in prompt
        Path(output).mkdir(mode=0o700)
        return Generation("Please meet at 11:30.", {"completion": "eos"}, Path(output))
    kp.generate = generate
    result = kp.rewrite_openai(chat("Meet at 10:30."), output=tmp_path / "rewrite")
    assert result.original == "Meet at 10:30."
    assert result.text == "Please meet at 11:30."
    assert result.checks["numbers_preserved"] is False
    assert result.status == "failed_checks"
    saved = json.loads((result.generation.artifacts / "rewrite.json").read_text())
    assert saved["original"] == result.original and saved["candidate"] == result.text
    assert saved["hosted_provider_watermark"] is False


def test_review_required_even_when_lexical_checks_pass():
    from keyprint import Rewrite
    checks = fidelity_checks("Do not pay 12 dollars.", "Pay 12 dollars.", completion="eos")
    result = Rewrite("Do not pay 12 dollars.", "Pay 12 dollars.", None, checks)
    assert result.status == "needs_review"  # Deliberate negation loss is invisible to lexical checks.
    assert fidelity_checks("Hello\nWorld", "Hello\\nWorld", completion="eos")["no_new_escaped_line_breaks"] is False
    assert fidelity_checks("Hello world", "Hello\\nworld", completion="eos")["no_new_escaped_line_breaks"] is False
    unchanged = fidelity_checks("Hello world", "Hello  world", completion="eos")
    assert Rewrite("Hello world", "Hello  world", None, unchanged).status == "failed_checks"
