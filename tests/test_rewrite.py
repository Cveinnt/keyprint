import json

import pytest

from keyprint import Keyprint, RewriteUnavailableError
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


@pytest.mark.parametrize("candidate", [
    '"Hi Maya, please review the draft by Friday at 09:30. Do not publish it before I approve it."',
    '“Hi Maya, please review the draft by Friday at 09:30. Do not publish it before I approve it.”',
    "HI MAYA, PLEASE REVIEW THE DRAFT BY FRIDAY AT 09:30. DO NOT PUBLISH IT BEFORE I APPROVE IT.",
])
def test_cosmetic_changes_are_not_a_successful_paraphrase(candidate):
    from keyprint.rewrite import Rewrite
    original = "Hi Maya, please review the draft by Friday at 09:30. Do not publish it before I approve it."
    checks = fidelity_checks(original, candidate, completion="eos")
    assert checks["changed"] and checks["canonical_changed"]
    assert checks["word_sequence_changed"] is False
    assert Rewrite(original, candidate, None, checks).status == "failed_checks"


def test_unicode_words_are_compared_without_erasing_accents():
    original = "Écrivez à Maya demain."
    assert fidelity_checks(original, "« Écrivez à Maya demain. »", completion="eos")["word_sequence_changed"] is False
    assert fidelity_checks(original, "Écrivez à Maya aujourd'hui.", completion="eos")["word_sequence_changed"] is True


@pytest.mark.parametrize("before,after", [
    ("Review the draft by Friday at 09:30.", '"Review the draft by Monday at 09:30."'),
    ("Répondez vendredi à 09:30.", "Veuillez répondre lundi à 09:30."),
    ("Revisa el texto el miércoles.", "Por favor, revisa el texto el sábado."),
    ("Monday or Monday again.", "Monday or another day."),
])
def test_changed_weekday_blocks_rewrite_even_when_numbers_match(before, after):
    from keyprint import Rewrite
    checks = fidelity_checks(before, after, completion="eos")
    assert checks["numbers_preserved"] is True
    assert checks["weekday_names_preserved"] is False
    assert Rewrite(before, after, None, checks).status == "failed_checks"


def test_weekday_check_allows_case_but_does_not_claim_event_alignment():
    assert fidelity_checks("Meet Friday.", "Please meet FRIDAY.", completion="eos")["weekday_names_preserved"]
    assert fidelity_checks("Read Sundayish.", "Read something else.", completion="eos")["weekday_names_preserved"]
    reordered = fidelity_checks("Depart Monday, return Friday.", "Depart Friday, return Monday.", completion="eos")
    assert reordered["weekday_names_preserved"] is True
    assert reordered["meaning_preservation"] == "not_measured"






def test_review_required_even_when_lexical_checks_pass():
    from keyprint import Rewrite
    checks = fidelity_checks("Do not pay 12 dollars.", "Pay 12 dollars.", completion="eos")
    result = Rewrite("Do not pay 12 dollars.", "Pay 12 dollars.", None, checks)
    assert result.status == "needs_review"  # Deliberate negation loss is invisible to lexical checks.
    assert fidelity_checks("Hello\nWorld", "Hello\\nWorld", completion="eos")["no_new_escaped_line_breaks"] is False
    assert fidelity_checks("Hello world", "Hello\\nworld", completion="eos")["no_new_escaped_line_breaks"] is False
    unchanged = fidelity_checks("Hello world", "Hello  world", completion="eos")
    assert Rewrite("Hello world", "Hello  world", None, unchanged).status == "failed_checks"


@pytest.mark.parametrize("candidate", ["", " \n\t"])
def test_empty_completed_candidate_fails(candidate):
    from keyprint import Rewrite
    checks = fidelity_checks("Please keep the backup.", candidate, completion="eos")
    assert checks["nonempty"] is False
    assert Rewrite("Please keep the backup.", candidate, None, checks).status == "failed_checks"


@pytest.mark.parametrize("preserve", [None, "Maya", {"Maya"}, [None], [""], [" "],
    ["missing"], ["Maya", "Maya"], ["Maya"] * 33, ["a" * 257]])
def test_invalid_protected_phrases_fail_before_inference(tmp_path, preserve):
    kp = Keyprint(key=bytes(range(32)))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid preservation settings reached generation")
    kp.generate = forbidden
    with pytest.raises(ValueError):
        kp.rewrite("Ask Maya to review the draft.", preserve=preserve, output=tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize("original,candidate,phrases", [
    ("Please ask Maya to read it.", "Kindly ask Ana to read it.", ["Maya"]),
    ("Move from Tuesday at 09:30 to Wednesday at 14:00.",
     "Please move from Wednesday at 09:30 to Tuesday at 14:00.", ["Tuesday at 09:30", "Wednesday at 14:00"]),
    ("Maya should ask Maya.", "Please ask Maya.", ["Maya"]),
    ("Meet Maya.", "Please ask Maya, Maya.", ["Maya"]),
    ("Écrivez à Élodie vendredi à 09:30.", "Contactez Elodie vendredi à 09:30.", ["Élodie"]),
    ("请于周四下午14:00联系王琳。", "请在周五下午14:00联系王琳。", ["周四下午14:00", "王琳"]),
])
def test_changed_phrases_fail_even_when_other_literals_match(original, candidate, phrases):
    from keyprint import Rewrite
    checks = fidelity_checks(original, candidate, completion="eos", preserve=phrases)
    assert checks["numbers_preserved"] and checks["weekday_names_preserved"]
    assert checks["protected_literals_preserved"] is False
    assert Rewrite(original, candidate, None, checks).status == "failed_checks"


def test_exact_phrases_do_not_claim_semantic_fidelity():
    from keyprint import Rewrite
    original = "Do not send Maya the draft by Friday at 09:30."
    candidate = "Send Maya the draft by Friday at 09:30."
    checks = fidelity_checks(original, candidate, completion="eos", preserve=["Maya", "Friday at 09:30"])
    assert checks["protected_literals_preserved"] is True
    assert checks["meaning_preservation"] == "not_measured"
    assert Rewrite(original, candidate, None, checks).status == "needs_review"




@pytest.mark.parametrize("provider", ["rewrite", "rewrite_openai", "rewrite_anthropic", "module"])
@pytest.mark.parametrize("source", [
    "Bonjour Maya, merci de relire le document. Ne le publiez pas avant mon approbation.",
    "Do not publish unless I approve. An acknowledgement is not approval.",
    "No publiques el documento sin mi aprobación.",
    "未经我批准，请勿发布。",
    "Bonjour Maya. Keep the quoted English phrase unchanged.",
])
def test_rewriting_is_blocked_without_inference_or_source_mutation(tmp_path, provider, source):
    import copy
    from keyprint.rewrite import rewrite
    kp = Keyprint(key=bytes(range(32)))
    def forbidden(*args, **kwargs):
        pytest.fail("unvalidated rewrite reached inference")
    kp.generate = forbidden
    value = (chat(source) if provider == "rewrite_openai" else
             claude(content=[{"type": "text", "text": source}]) if provider == "rewrite_anthropic" else source)
    before = copy.deepcopy(value)
    with pytest.raises(RewriteUnavailableError, match="No text was rewritten"):
        if provider == "module":
            rewrite(kp, source, output=tmp_path / "attempt")
        else:
            getattr(kp, provider)(value, output=tmp_path / "attempt")
    assert value == before
    assert not (tmp_path / "attempt").exists()
