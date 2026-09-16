"""Explicit binding extension; never mutates a frozen study binding."""
import copy

from .grouped_canonical_prototype import Config, Profile
from .token_source_policy import TokenSourceBinding
from .token_runtime_profile import RuntimeBoundProfile

SCORE_NAMESPACE = "690971bdf929e35ecb134916c5ff4b5878562114c802d917face52fc7ee75507"
RUNTIME_LIMITS = (1024, 2048)


def runtime_binding(*, max_steps, base=None):
    if type(max_steps) is not int or max_steps not in RUNTIME_LIMITS:
        raise ValueError("runtime bound must be 1024 or 2048")
    base = TokenSourceBinding() if base is None else base
    if (type(base) is not TokenSourceBinding or type(base.profile) is not Profile
            or base.profile.digest != SCORE_NAMESPACE or base.profile.config != Config(max_steps=512)):
        raise ValueError("original frozen token/source binding required")
    result = copy.copy(base)
    result._profile = RuntimeBoundProfile(base.profile, max_steps=max_steps)
    validate_runtime_binding(result)
    return result


def validate_runtime_binding(binding):
    if type(binding) is not TokenSourceBinding or type(binding.profile) is not RuntimeBoundProfile:
        raise ValueError("explicit runtime-bound token/source binding required")
    p = binding.profile
    if p.config.max_steps not in RUNTIME_LIMITS or p.score_namespace_sha256 != SCORE_NAMESPACE:
        raise ValueError("unsupported runtime/score namespace")
    original = Profile(binding.token_bytes, tokenizer_identity=p.tokenizer_identity, config=Config(max_steps=512))
    expected = RuntimeBoundProfile(original, max_steps=p.config.max_steps)
    if (original.digest != SCORE_NAMESPACE or p.digest != expected.digest
            or p._domain != original._domain or p.classes != original.classes
            or p.config != expected.config or p.original_max_steps != 512):
        raise ValueError("runtime binding identity mismatch")
    return p.identity_receipt()
