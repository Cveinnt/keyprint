"""Research-only prompt-free working likelihood ratio, not an SDK verdict.

Fixed continuation probabilities estimate the unmarked distribution. A 50/50
mixture of that distribution and its keyed tournament supplies the numerator.
Scoring never reads the original request, generation trace or future model head.
The ideal-random-PRF normalization argument is not fixed-key calibration, a
floating-point proof, an authorship probability or an anytime-valid guarantee.
"""
import hashlib
import math

import numpy as np

from predictability_filter import PREFIX, CHUNK_SIZE, fixed_batch, causal_inputs
from keyprint.sampling import sparse_softmax
from keyprint._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter
from log_tournament import token_log_ratio, mixture_log_factor

MIXTURE = .5
LOG_CUTOFF = math.log(200.)


def measure(backend, binding, text):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    ids = list(binding.encode_visible(text))
    if not ids or len(ids) > binding.profile.config.max_steps:
        raise ValueError("Nonempty bounded text tokens required")
    prefix = backend.encode_prompt(PREFIX)
    warmup, inputs = causal_inputs(prefix, ids)
    cache = make_prompt_cache(backend.model)
    if warmup:
        backend.model(mx.array([warmup], dtype=mx.int32), cache=cache)
        mx.eval([entry.state for entry in cache])
    rows = []
    for offset in range(0, len(inputs), CHUNK_SIZE):
        chunk = inputs[offset:offset + CHUNK_SIZE]
        raw = backend.model(mx.array([fixed_batch(chunk)], dtype=mx.int32), cache=cache)
        raw = raw[0, :len(chunk)].astype(mx.float32)
        mx.eval(raw)
        for head in np.array(raw):
            filtered = stable_support_filter(head[None], temperature=.7, top_k=100,
                                             mapped_vocabulary_size=len(binding.profile.classes))
            p = sparse_softmax(filtered.filtered_logits[0])[:len(binding.profile.classes)]
            support = np.flatnonzero(p > 0)
            rows.append({"raw_head_sha256": hashlib.sha256(head.tobytes()).hexdigest(),
                         "support": list(map(int, support)), "probabilities": p[support].tolist()})
    if len(rows) != len(ids):
        raise ValueError("Predictive head count differs from text token count")
    return {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "token_ids": ids,
            "fixed_prefix_ids": prefix, "heads": rows,
            "original_prompt_used": False, "key_used_for_model_inference": False}


def dense_head(row, width):
    support, values = row["support"], row["probabilities"]
    if (not support or len(support) > 100 or len(support) != len(values)
            or any(type(i) is not int or not 0 <= i < width for i in support)
            or support != sorted(set(support))):
        raise ValueError("Invalid sparse support")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Invalid sparse probability")
    if abs(math.fsum(values) - 1.) > 1e-12:
        raise ValueError("Sparse distribution is not normalized")
    p = np.zeros(width, dtype=np.float64)
    p[support] = values
    return p


def score(profile, key, measurement):
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("32-byte key required")
    ids, heads = measurement["token_ids"], measurement["heads"]
    if (not ids or len(ids) != len(heads) or len(ids) > profile.config.max_steps
            or any(type(i) is not int or not 0 <= i < len(profile.classes) for i in ids)):
        raise ValueError("Exact bounded token/head alignment required")
    context, used, terms = (), set(), []
    for index, (token, head) in enumerate(zip(ids, heads, strict=True)):
        p = dense_head(head, len(profile.classes))
        label = profile.classes[token]
        reason, contribution = "scored", 0.
        base, marked_log = float(p[token]), None
        if label is None:
            reason = "excluded_label"
        elif context in used:
            reason = "repeated_context"
        elif base == 0:
            reason = "outside_surrogate_support"
        else:
            ratio = token_log_ratio(p, profile, key, context, token)
            marked_log = math.log(base) + ratio
            contribution = mixture_log_factor(ratio, MIXTURE)
        terms.append({"index": index, "token": token, "reason": reason, "base": base,
                      "marked_log_probability": marked_log, "log_ratio": contribution})
        # Even an out-of-support observation advances the literal context.
        # Duplicate contexts never provide a second independent contribution.
        if label is not None:
            used.add(context)
            context = (*context, label)[-profile.config.history:]
    value = math.fsum(t["log_ratio"] for t in terms)
    return {"working_log_ratio": value, "flagged": value >= LOG_CUTOFF,
            "terms": terms, "scored_events": sum(t["reason"] == "scored" for t in terms),
            "outside_support": sum(t["reason"] == "outside_surrogate_support" for t in terms),
            "calibrated": False}
