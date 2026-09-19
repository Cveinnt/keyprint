"""Development-only, key-blind position selection using token predictability.

Entropy uses only a fixed chat prefix and preceding literal text. Neither the
original request, the current token's probability nor watermark bits select a
position. A selected-bit reference tail still needs real null calibration.
"""
import hashlib
import math

import numpy as np

PREFIX = "Continue the text."
MIN_ENTROPY_BITS = .5
CHUNK_SIZE = 64
PAD_TOKEN_ID = 0


def entropy_bits(raw):
    from keyprint._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter
    filtered = stable_support_filter(raw, temperature=.7, top_k=100,
                                     mapped_vocabulary_size=151669).filtered_logits
    logits = filtered[np.isfinite(filtered)]
    weights = np.exp(logits)
    probabilities = weights / math.fsum(weights)
    value = -math.fsum(float(p) * math.log2(float(p)) for p in probabilities)
    if not math.isfinite(value) or value < 0:
        raise ArithmeticError("Invalid predictive entropy")
    return value


def causal_inputs(prefix_ids, text_ids):
    if not prefix_ids or not text_ids:
        raise ValueError("Nonempty prefix and text tokens required")
    # Warm the cache with prefix[:-1]. Every returned head predicts the next
    # text token; the current/future text token cannot condition its own head.
    return list(prefix_ids[:-1]), [prefix_ids[-1], *text_ids[:-1]]


def fixed_batch(values):
    if not 1 <= len(values) <= CHUNK_SIZE:
        raise ValueError("A nonempty bounded inference chunk is required")
    return [*values, *([PAD_TOKEN_ID] * (CHUNK_SIZE-len(values)))]


def eligible_positions(profile, token_ids):
    context, used, positions = (), set(), []
    for index, token in enumerate(token_ids):
        label = profile.classes[token]
        if label is None:
            continue
        if context not in used:
            positions.append(index)
        used.add(context)
        context = (*context, label)[-profile.config.history:]
    return positions


def selected_positions(eligible, entropies):
    if any(not math.isfinite(v) or v < 0 for v in entropies):
        raise ValueError("Finite nonnegative entropy required")
    if len(set(eligible)) != len(eligible) or any(type(i) is not int or not 0 <= i < len(entropies) for i in eligible):
        raise ValueError("Distinct eligible token positions required")
    return [i for i in eligible if entropies[i] >= MIN_ENTROPY_BITS]


def measure(backend, binding, text):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    ids = binding.encode_visible(text)
    if not ids:
        raise ValueError("Empty literal token sequence")
    prefix = backend.encode_prompt(PREFIX)
    warmup, inputs = causal_inputs(prefix, ids)
    cache = make_prompt_cache(backend.model)
    if warmup:
        backend.model(mx.array([warmup], dtype=mx.int32), cache=cache)
        mx.eval([entry.state for entry in cache])
    values, head_hashes = [], []
    for offset in range(0, len(inputs), CHUNK_SIZE):
        chunk = inputs[offset:offset+CHUNK_SIZE]
        # Quantized kernels can differ by batch shape. Fix every text batch to
        # 64 positions, discarding padded future heads. Only the final batch is
        # padded; no real token ever observes padding in its causal prefix.
        raw = backend.model(mx.array([fixed_batch(chunk)], dtype=mx.int32), cache=cache)
        raw = raw[0, :len(chunk)].astype(mx.float32)
        mx.eval(raw)
        raw = np.array(raw)
        if len(raw) != len(chunk):
            raise ValueError("Model head count differs from input count")
        for head in raw:
            head_hashes.append(hashlib.sha256(head.tobytes()).hexdigest())
            values.append(entropy_bits(head[None]))
    eligible = eligible_positions(binding.profile, ids)
    return {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids": list(ids), "entropy_bits": values,
            "selected_positions": selected_positions(eligible, values),
            "eligible_positions": eligible, "raw_head_sha256": head_hashes,
            "fixed_prefix_ids": prefix, "original_prompt_used": False,
            "watermark_key_used": False, "new_generation": False}


def selected_bits(binding, text, key, selection):
    if hashlib.sha256(text.encode()).hexdigest() != selection["text_sha256"]:
        raise ValueError("Selection belongs to another text")
    replay = binding.replay_text(text, key)
    if list(replay.token_ids) != selection["token_ids"]:
        raise ValueError("Selection tokenization differs")
    eligible = eligible_positions(binding.profile, replay.token_ids)
    if eligible != selection["eligible_positions"]:
        raise ValueError("Selection eligibility differs")
    selected = selected_positions(eligible, selection["entropy_bits"])
    if selected != selection["selected_positions"]:
        raise ValueError("Selection differs from frozen entropy rule")
    if len(selection["entropy_bits"]) != len(replay.token_ids):
        raise ValueError("Entropy count differs from token count")
    # Replay emits one event for each nonempty canonical label, including
    # duplicates. Pair those with literal token positions before filtering.
    positions = [i for i,t in enumerate(replay.token_ids) if binding.profile.classes[t] is not None]
    chosen = set(selected)
    rows = [event.bits for i,event in zip(positions, replay.events, strict=True) if i in chosen]
    return np.asarray(rows, dtype=np.int8).reshape((-1, binding.profile.config.layers))
