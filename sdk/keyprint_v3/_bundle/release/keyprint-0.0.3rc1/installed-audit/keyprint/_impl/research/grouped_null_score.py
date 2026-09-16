"""Research random-key null statistic. Not an authorship or deployment verdict."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class NullScore:
    events: int
    trials: int
    ones: int
    random_key_null_p: float | None
    underflow_floor: bool


def tail(ones: int, trials: int):
    if type(ones) is not int or type(trials) is not int or not 0 <= ones <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    p = float(binom.sf(ones - 1, trials, .5))
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ArithmeticError("invalid binomial tail")
    return max(p, float(np.nextafter(0., 1.))), p == 0.


def score(events, layers: int):
    if type(layers) is not int or layers < 1:
        raise ValueError("invalid layer count")
    seen, n, total = set(), 0, 0
    for event in events:
        if not event.eligible:
            if event.bits is not None:
                raise ValueError("ineligible event contains scored bits")
            continue
        if event.context in seen:
            raise ValueError("repeated eligible context invalidates independent-bit null")
        if not event.label or event.bits is None or len(event.bits) != layers:
            raise ValueError("invalid eligible event")
        if any(type(bit) is not int or bit not in (0, 1) for bit in event.bits):
            raise ValueError("nonbinary event bits")
        seen.add(event.context)
        n += 1
        total += sum(event.bits)
    if not n:
        return NullScore(0, 0, 0, None, False)
    p, floor = tail(total, n * layers)
    return NullScore(n, n * layers, total, p, floor)
