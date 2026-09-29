"""Exact small-support signal analysis for the bounded research oracle.

Ideal independent fair labels, one prefix. Not a detector or model-quality study.
Future oracle updates are martingales, so the expected final probability of a
token times an already observed label equals its expectation just after that
layer. This lets us sum expected matching labels without enumerating token draws.
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import argparse
import json
from pathlib import Path

from predictable_budget_oracle import predictable_strength, step


def analyze(base, *, layers=30, max_states=1024, policy='maximum'):
    if policy == 'maximum':
        strength, update = predictable_strength, step
    elif policy == 'paced-eighth':
        from paced_budget_oracle import paced_strength, step as paced_step
        strength, update = paced_strength, paced_step
    else:
        raise ValueError('Unknown exact allocation policy')
    if type(layers) is not int or not 1 <= layers <= 30:
        raise ValueError('Require one to thirty layers')
    if not 1 <= len(base) <= 3 or type(max_states) is not int or max_states < 1:
        raise ValueError('Bounded small-support analysis only')
    base = tuple(base)
    predictable_strength(base, base)  # Exact normalization/type/bound validation.
    labels = tuple(product((0, 1), repeat=len(base)))
    probability = F(1, len(labels))
    states = {base: F(1)}
    expected_matches = F(0)
    expected_active = F(0)
    trace = []
    for layer in range(layers):
        following = defaultdict(F)
        layer_matches, active = F(0), F(0)
        for q, mass in states.items():
            if strength(base, q) > 0: active += mass
            for bits in labels:
                out, _ = update(base, q, bits)
                weight = mass * probability
                following[out] += weight
                # Predictable bounded updates preserve conditional expectation;
                # subsequent layers cannot change this term's expected value.
                layer_matches += weight * sum(p*g for p, g in zip(out, bits))
                if len(following) > max_states:
                    raise ValueError('State budget exceeded; do not approximate or drop states')
        states = dict(following)
        if sum(states.values()) != 1:
            raise AssertionError('State mass lost')
        marginal = tuple(sum(mass*q[i] for q, mass in states.items()) for i in range(len(base)))
        if marginal != base:
            raise AssertionError('Exact random-label mean changed')
        expected_matches += layer_matches
        expected_active += active
        trace.append({'layer': layer+1, 'states': len(states),
            'probability_strength_positive': active,
            'expected_matching_bit': layer_matches})
    frozen = sum(mass for q, mass in states.items() if strength(base, q) == 0)
    return {'layers': layers, 'base': base, 'ratio': F(2),
        'expected_matching_bit_fraction': expected_matches/layers,
        'ordinary_or_independent_key_bit_fraction': F(1, 2),
        'expected_layers_with_positive_strength': expected_active,
        'probability_strength_exhausted_after_last_layer': frozen,
        'exact_final_marginal': marginal, 'trace': trace}


def serializable(value):
    if isinstance(value, F):
        return {'exact': str(value), 'decimal': float(value)}
    if isinstance(value, dict): return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(x) for x in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixtures = [(F(1,2), F(1,2)), (F(3,4), F(1,4)),
                (F(99,100), F(1,100)), (F(999,1000), F(1,1000))]
    result = {'schema': 'keyprint.predictable-budget-capacity.v1',
        'selection': 'Four fixed two-token mathematical fixtures; all label assignments included by exact state propagation',
        'rows': [serializable(analyze(base)) for base in fixtures],
        'scope': 'Ideal independent fair bits at one prefix; no finite-key, grouped-vocabulary, sequence, power or quality claim',
        'model_inference': False, 'detector_calibrated': False, 'quality_acceptance': False}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')


if __name__ == '__main__': main()
