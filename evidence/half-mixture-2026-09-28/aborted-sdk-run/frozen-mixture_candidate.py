"""Research-only half mixture, outside the installable SDK and default API.

Shares the reference PRF namespace for a controlled comparison, but changes the
sampling law and runtime identity. No detector or quality acceptance transfers.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from keyprint.backends.mlx_bounded import BoundedReferenceCandidate
from keyprint.experimental.capped_utf8 import CappedPipeline
from keyprint.experimental.fast_public import FastPublicCandidate
from keyprint._engine.research.keyprint_candidate_v3.adapter import V3Host, SourceRequest, ChannelRequest, digest
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Prepared
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession

VERSION = 'keyprint-research-half-mixture-v1'


def half_mixture(base, marked):
    # Sum before halving: two equal positive subnormals must not become zero.
    result = (base + marked) * .5
    if (not np.isfinite(result).all() or (result < 0).any()
            or not np.array_equal(result > 0, base > 0)
            or abs(math.fsum(result[result > 0]) - 1.) > 1e-12):
        raise ArithmeticError('Invalid half-mixture distribution')
    return result


class MixtureSession(SparseTokenSourceSession):
    def prepare(self, probabilities):
        reference = super().prepare(probabilities)
        if self._condition == 'ordinary':
            return reference
        mixed = half_mixture(probabilities, reference.probabilities)
        protected = self._last_decision['protected_token_ids']
        if protected and not np.array_equal(mixed[list(protected)], probabilities[list(protected)]):
            raise ArithmeticError('Protected probability changed')
        self._pending = Prepared(reference.index, np.frombuffer(mixed.tobytes(), dtype=np.float64))
        self._support = mixed > 0
        return self._pending


class MixtureHost(V3Host):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self._visible.session
        if type(old) is not SparseTokenSourceSession or old._steps or old._pending is not None:
            raise RuntimeError('Research mixture requires a fresh reference session')
        replacement = MixtureSession(old.profile, old._key, condition=old._condition, request=old._request)
        old.close()
        self._visible.session = replacement


class MixtureCandidate(BoundedReferenceCandidate):
    def __init__(self, reference):
        super().__init__(reference)
        spec = self._identity['specification']
        spec['version'] = VERSION
        spec['sampling_change'] = {'law': '0.5 * (base + reference_marked)',
                                   'ordinary_unchanged': True,
                                   'reference_prf_namespace_retained': True,
                                   'empirical_results_transfer': False,
                                   'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        spec['execution']['sampling'] = 'research_half_mixture_v1'
        self._identity.update(version=VERSION, runtime_profile_sha256=digest(spec))

    def pipeline(self, key, *, condition, purpose='general', source_text=None,
                 allow_thinking=False, allow_tools=False):
        if purpose != 'general' or source_text is not None or allow_thinking or allow_tools:
            raise ValueError('Research half mixture supports plain generation only')
        self.reference._base._scorer.check_key(key)
        raw = MixtureHost(key, condition=condition, binding=self.reference._base._binding,
                         request=SourceRequest(), channels=ChannelRequest(False, False),
                         v2_identity=self.identity, filter_settings=self.filter_settings)
        return CappedPipeline(raw, hashlib.sha256(key).hexdigest())


class MixturePublicCandidate(FastPublicCandidate):
    core_type = MixtureCandidate
    facade_version = VERSION + '-facade'
    integration_status = 'research_changed_sampling_law_not_qualified'
    package_scope = 'Research half mixture; no transferred quality, detector or serving acceptance'
