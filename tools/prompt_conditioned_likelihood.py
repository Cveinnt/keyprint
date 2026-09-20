"""Research-only likelihood replay when the original prompt is available.

Reuse the frozen causal, fixed-batch surrogate measurement and score. Change
only its conditioning prompt. No generation journals, probabilities or draws
are inputs. This does not solve prompt-free detection or establish calibration.
"""
import hashlib

from surrogate_likelihood import measure as surrogate_measure, PREFIX


class PromptConditionedModel:
    """Replace only the fixed continuation prompt at the measurement boundary."""

    def __init__(self, backend, prompt):
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 16000:
            raise ValueError("Original prompt must contain 1 to 16000 characters")
        self.backend, self.prompt = backend, prompt
        self.model = backend.model

    def encode_prompt(self, placeholder):
        if placeholder != PREFIX:
            raise ValueError("Frozen measurement prefix differs")
        return self.backend.encode_prompt(self.prompt)


def measure(backend, binding, text, prompt):
    result = surrogate_measure(PromptConditionedModel(backend, prompt), binding, text)
    # The reused routine's fixed-prefix provenance must not survive this change.
    result["conditioning_prefix_ids"] = result.pop("fixed_prefix_ids")
    result["original_prompt_used"] = True
    result["original_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    result["private_generation_data_used"] = False
    result["scope"] = "Prompt-conditioned research replay; uncalibrated, not prompt-free"
    return result
