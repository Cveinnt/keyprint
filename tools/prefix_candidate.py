"""Research-only five-layer tournament in the reference's original PRF domain.

Fixed depth, not entropy-adaptive stopping or post-transform mixing. The original
30-layer namespace is intentionally retained for paired mechanism comparisons.
This is not a separately configured five-layer SDK profile or a calibrated score.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from keyprint.backends.mlx_bounded import BoundedReferenceCandidate
from keyprint.experimental.capped_utf8 import CappedPipeline
from keyprint._engine.research.keyprint_candidate_v3.adapter import V3Host, SourceRequest, ChannelRequest, digest
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Prepared
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession, bit_table
from keyprint._engine.legacy._impl.research.byte_trie_numeric import MIN_POSITIVE, update

LAYERS = 5
VERSION = 'keyprint-research-prefix-five-v1'


def transform(q, profile, key, context, protected, counters):
    ids = [int(i) for i in np.flatnonzero(q > 0)
           if int(i) not in protected and profile.classes[int(i)] is not None]
    labels = dict.fromkeys(profile.classes[i] for i in ids)
    out = q.copy()
    if len(labels) < 2:
        return out
    mass = math.fsum(float(q[i]) for i in ids)
    weights = tuple(float(q[i]) / mass for i in ids)
    table = bit_table(profile, key, context, labels)
    for layer in range(LAYERS):
        weights = update(weights, [table[profile.classes[i]][layer] for i in ids], counters)
    for i, p in zip(ids, weights):
        value = mass * p
        if value == 0.:
            value = MIN_POSITIVE
            counters['partition_roundups'] += 1
        out[i] = value
    return out


class PrefixSession(SparseTokenSourceSession):
    def prepare(self, probabilities):
        # Reuse all reference validation/state transitions, then replace only the
        # active transform. Discard diagnostics from the unused 30-layer pass.
        active = self._condition == 'marked' and self._context not in self._used
        counters = self._numeric_counters.copy()
        reference = super().prepare(probabilities)
        decision = self._last_decision
        if not active or decision['mode'] == 'startup_ordinary':
            return reference
        self._numeric_counters.clear()
        self._numeric_counters.update(counters)
        protected = decision['protected_token_ids']
        out = transform(probabilities, self.profile, self._key, self._context,
                        protected, self._numeric_counters)
        if (not np.isfinite(out).all() or (out < 0).any()
                or abs(math.fsum(out[out > 0]) - 1.) > 1e-12
                or not np.array_equal(out > 0, probabilities > 0)):
            raise ArithmeticError('Invalid prefix distribution or support')
        if protected and not np.array_equal(out[list(protected)], probabilities[list(protected)]):
            raise ArithmeticError('Protected probability changed')
        self._pending = Prepared(reference.index, np.frombuffer(out.tobytes(), dtype=np.float64))
        self._support = out > 0
        return self._pending


class PrefixHost(V3Host):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self._visible.session
        if type(old) is not SparseTokenSourceSession or old._steps or old._pending is not None:
            raise RuntimeError('Research prefix requires a fresh reference session')
        replacement = PrefixSession(old.profile, old._key, condition=old._condition, request=old._request)
        old.close()
        self._visible.session = replacement


class PrefixCandidate(BoundedReferenceCandidate):
    def __init__(self, reference):
        super().__init__(reference)
        spec = self._identity['specification']
        spec['version'] = VERSION
        spec['sampling_change'] = {'law': 'first five tournament layers',
                                   'ordinary_unchanged': True, 'layers_executed': LAYERS,
                                   'reference_prf_namespace_retained': True,
                                   'empirical_results_transfer': False,
                                   'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        spec['execution']['sampling'] = 'research_prefix_five_v1'
        self._identity.update(version=VERSION, runtime_profile_sha256=digest(spec))

    def pipeline(self, key, *, condition, purpose='general', source_text=None,
                 allow_thinking=False, allow_tools=False):
        if purpose != 'general' or source_text is not None or allow_thinking or allow_tools:
            raise ValueError('Research prefix supports plain generation only')
        self.reference._base._scorer.check_key(key)
        raw = PrefixHost(key, condition=condition, binding=self.reference._base._binding,
                         request=SourceRequest(), channels=ChannelRequest(False, False),
                         v2_identity=self.identity, filter_settings=self.filter_settings)
        return CappedPipeline(raw, hashlib.sha256(key).hexdigest())


def prefix_tail(bits):
    """Exact binomial sum, converted to float; ideal fresh/fair null bits only."""
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] != 30 or not 1 <= len(bits) <= 2048 or not np.isin(bits, [0, 1]).all():
        raise ValueError('Require 1-2048 binary events with 30 namespace layers')
    successes = int(bits[:, :LAYERS].sum())
    trials = len(bits) * LAYERS
    # Rational integer arithmetic before one final float conversion.
    numerator = sum(math.comb(trials, i) for i in range(successes, trials + 1))
    value = numerator / (1 << trials)
    return {'reference_tail': min(1., math.nextafter(value, math.inf)),
            'successes': successes, 'trials': trials, 'events': len(bits),
            'scope': 'Ideal fair-bit null; fixed-key deployment calibration unestablished'}
