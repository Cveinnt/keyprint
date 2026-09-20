"""Development-only fixed layer weighting of archived model-centered residuals.

Weights decrease linearly from 1 to .1, following the relative 10-to-1 layer
weights in SynthID Text's weighted mean. This is not its detector implementation.
Five fixed whole-document betting processes are averaged, never maximized.
For fixed key-independent text/predictions and ideal fresh independent PRF
addresses, each factor has mean one. That argument is not fixed-key deployment
calibration, a floating-point guarantee or evidence about adversarial inputs.
"""
import math

STAKES = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)
LOG_CUTOFF = math.log(200.)


def weights(layers):
    if type(layers) is not int or not 2 <= layers <= 256:
        raise ValueError("At least two bounded layers required")
    return tuple((10 * (layers - 1) - 9 * i) / (10 * (layers - 1))
                 for i in range(layers))


def aggregate(terms, *, layers=30):
    layer_weights = weights(layers)
    values = []
    for term in terms:
        residuals = term["residuals"]
        if term["reason"] not in ("scored", "excluded_label", "repeated_context", "outside_surrogate_support"):
            raise ValueError("Unknown evidence eligibility")
        if len(residuals) != (layers if term["reason"] == "scored" else 0):
            raise ValueError("Residual count differs from eligibility or layer count")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1 for v in residuals):
            raise ValueError("Finite residuals in [-1, 1] required")
        values.extend(w * r for w, r in zip(layer_weights, residuals))
    components = [math.fsum(math.log1p(stake * value) for value in values) for stake in STAKES]
    maximum = max(components)
    evidence = maximum + math.log(math.fsum(math.exp(v - maximum) for v in components) / len(STAKES))
    return {"component_log_evidence": components, "working_log_evidence": evidence,
            "flagged": evidence >= LOG_CUTOFF, "calibrated": False}
