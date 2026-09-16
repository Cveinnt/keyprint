"""Float64 trie with cancellation-free tournament complements.

Separate numerical implementation. The September 5 exact prototype is frozen.
Positive weights that round to zero are rounded up to float64's least positive
subnormal. This local numerical policy is counted, versioned, and not exact algebra.
"""
from fractions import Fraction
import math

from .byte_trie_tournament import State, canonical, transform as exact_transform

KERNEL_VERSION = "complement-fsum-subnormal-v2"
MIN_POSITIVE = math.ulp(0.)


def update(q, bits, diagnostics=None):
    if not q or len(q) != len(bits) or any(type(g) is not int or g not in (0, 1) for g in bits):
        raise ValueError("invalid tournament inputs")
    if any(not math.isfinite(p) or p < 0 for p in q):
        raise ValueError("invalid tournament probability")
    total = math.fsum(q)
    if total <= 0 or abs(total - 1.) > 1e-12:
        raise ValueError("tournament probabilities must sum to one")
    # q'_i = q_i * (1 + g_i - mean(g)). For g=0, evaluate the
    # complementary mass directly instead of subtracting nearly equal numbers.
    # With normalized q this is exactly q_i * (mass(g=0) + g_i).
    zero_mass = math.fsum(p for p, g in zip(q, bits) if g == 0)
    weights = tuple(p * (zero_mass + g * total) for p, g in zip(q, bits))
    denominator = math.fsum(weights)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ArithmeticError("invalid tournament normalization")
    output = tuple(p / denominator for p in weights)
    floored = sum(before > 0 and after == 0 for before, after in zip(q, output))
    if floored:
        output = tuple(MIN_POSITIVE if before > 0 and after == 0 else after
                       for before, after in zip(q, output))
        if diagnostics is not None:
            diagnostics["branch_roundups"] = diagnostics.get("branch_roundups", 0) + floored
    return output


def transform(probabilities, pieces, state, config, bits, diagnostics=None):
    if probabilities and all(isinstance(p, Fraction) for p in probabilities.values()):
        return exact_transform(probabilities, pieces, state, config, bits)
    if not isinstance(state, State) or len(state.context) > config.history:
        raise ValueError("invalid byte-trie state")
    if not probabilities or set(probabilities) - set(pieces):
        raise ValueError("every candidate needs an explicit byte mapping")
    if any(type(i) is not int or i < 0 for i in probabilities):
        raise ValueError("candidate IDs must be nonnegative integers")
    if any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
        raise ValueError("invalid base weights")
    total = math.fsum(probabilities.values())
    if abs(total - 1.) > 1e-12:
        raise ValueError("base weights must sum to one")
    labels = {i: canonical(pieces[i]) for i in probabilities}
    output = dict.fromkeys(probabilities, 0.)

    def visit(ids, depth, mass, context, used):
        node_mass = math.fsum(probabilities[i] for i in ids)
        children = {}
        for i in ids:
            if len(labels[i]) == depth:
                output[i] = mass * (probabilities[i] / node_mass)
            else:
                children.setdefault(labels[i][depth], []).append(i)
        if not children:
            return
        byte_values = sorted(children)
        child_mass = [math.fsum(probabilities[i] for i in children[b]) for b in byte_values]
        continuation = math.fsum(child_mass)
        conditional = tuple(p / continuation for p in child_mass)
        if context not in used and len(byte_values) > 1:
            for layer in range(config.layers):
                conditional = update(conditional, [bits(context, b, layer) for b in byte_values], diagnostics)
        next_used = used | {context}
        for byte, p in zip(byte_values, conditional):
            visit(children[byte], depth + 1, mass * (continuation / node_mass) * p,
                  (context + bytes((byte,)))[-config.history:], next_used)

    visit([i for i, p in probabilities.items() if p > 0], 0, total, state.context, state.used)
    if (any(p < 0 or not math.isfinite(p) for p in output.values())
            or abs(math.fsum(output.values()) - 1.) > 1e-10):
        raise ArithmeticError("trie output is not a distribution")
    floored = sum(probabilities[i] > 0 and p == 0 for i, p in output.items())
    if floored:
        output = {i: MIN_POSITIVE if probabilities[i] > 0 and p == 0 else p
                  for i, p in output.items()}
        if diagnostics is not None:
            diagnostics["token_roundups"] = diagnostics.get("token_roundups", 0) + floored
    return output
