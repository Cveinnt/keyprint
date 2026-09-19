"""Research conditional-layer density; not an SDK detector or calibrated verdict.

Motivated by SynthID Text's autoregressive layer likelihood, independently
implemented with NumPy/SciPy. Fit only on development marked text. Each binary
conditional sums to one. Under independent fair null bits, the resulting
likelihood ratio has expectation one. Real fixed-key deployment needs separate
validation; this ideal-PRF argument is not a measured error-rate guarantee.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

REFERENCE = "https://github.com/google-deepmind/synthid-text/blob/addb4a158143c7c6851a1308f78b89fceed59683/src/synthid_text/detector_bayesian.py"
FRACTIONS = (.05, .1, .25, .5, 1.)
RIDGE = .01


def binary_matrix(bits):
    bits = np.asarray(bits, dtype=float)
    if bits.ndim != 2 or not bits.shape[0] or not bits.shape[1] or not np.isin(bits, [0, 1]).all():
        raise ValueError("Nonempty binary event-by-layer matrix required")
    return bits


def extract(binding, text, key):
    events = binding.replay_text(text, key).events
    eligible = [e for e in events if e.eligible]
    if len({e.context for e in eligible}) != len(eligible):
        raise ValueError("Repeated eligible contexts invalidate the null factorization")
    return binary_matrix([e.bits for e in eligible])


def fit(bits):
    bits = binary_matrix(bits)
    parameters, diagnostics = [], []
    for layer in range(bits.shape[1]):
        x = np.column_stack([np.ones(len(bits)), bits[:, :layer] - .5])
        y = bits[:, layer]

        def objective(theta):
            s = expit(x @ theta)
            p = .5 + .25 * s
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
            derivative = (p - y) / (p * (1 - p)) * .25 * s * (1 - s)
            gradient = x.T @ derivative / len(y)
            gradient[1:] += RIDGE * theta[1:]
            return loss + RIDGE * float(theta[1:] @ theta[1:]) / 2, gradient

        initial = np.zeros(layer + 1); initial[0] = -2.5
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=[(-12, 12)] * (layer + 1),
                          options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f"Layer {layer} fit did not converge")
        parameters.append(result.x.tolist())
        diagnostics.append({"layer": layer, "iterations": int(result.nit), "loss": float(result.fun)})
    return {"parameters": parameters, "fractions": list(FRACTIONS), "ridge": RIDGE,
            "fit_events": len(bits), "diagnostics": diagnostics,
            "scope": "Opened-data research density, not a calibrated detector", "reference": REFERENCE}


def event_log_ratios(bits, model):
    bits = binary_matrix(bits)
    parameters = model["parameters"]
    if len(parameters) != bits.shape[1]:
        raise ValueError("Detector depth does not match bit depth")
    total = np.zeros(len(bits))
    for layer, theta in enumerate(parameters):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (layer + 1,) or not np.isfinite(theta).all():
            raise ValueError("Invalid conditional-layer coefficients")
        p = .5 + .25 * expit(theta[0] + (bits[:, :layer] - .5) @ theta[1:])
        total += np.where(bits[:, layer] == 1, np.log(2 * p), np.log(2 * (1 - p)))
    return total


def log_evidence(bits, model):
    ratios = event_log_ratios(bits, model)
    fractions = np.asarray(model["fractions"], dtype=float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or (fractions <= 0).any() or (fractions > 1).any():
        raise ValueError("Mixture fractions must be finite, nonempty and in (0, 1]")
    values = []
    for fraction in fractions:
        # Averaging whole-document evidence, never maximizing after seeing data.
        terms = ratios if fraction == 1 else np.logaddexp(np.log1p(-fraction), np.log(fraction) + ratios)
        values.append(float(terms.sum()))
    return float(logsumexp(values) - np.log(len(values)))
