"""V3 raw-model-head entry: shared support filtering, then keyed v2 engine.

This changes ordinary sampling, including mapping, numerical scaling, tail and
tie policies. V1/v2 study results do not transfer. Public entry accepts only raw
float32 model heads; there is no second public prefiltered-logits entry.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np

from keyprint_candidate_v2.adapter import (
    Candidate as V2Candidate, Pipeline as V2Pipeline, V2Host,
    SourceRequest, ChannelRequest, ROOT, digest,
)
import keyprint_stable_support_filter_v3 as support_filter

VERSION = 'keyprint-candidate-v3-shared-support-2026-09-09'


class Candidate(V2Candidate):
    def __init__(self, *, max_steps=2048, temperature=.7, top_k=100, max_logit_gap=600):
        # Validate the policy before allocating the larger inherited tokenizer.
        support_filter.stable_support_filter(np.zeros((1,1), dtype=np.float32),
            temperature=temperature, top_k=top_k, max_logit_gap=max_logit_gap)
        super().__init__(max_steps=max_steps)
        inherited = self.identity
        self._filter_settings = dict(temperature=float(temperature), top_k=top_k,
                                     max_logit_gap=float(max_logit_gap))
        policy = support_filter.filter_identity(**self._filter_settings,
            vocabulary_size=151936, mapped_vocabulary_size=151669)
        dependencies = dict(inherited['specification']['dependencies'])
        files = [Path(__file__), Path(__file__).with_name('__init__.py'), Path(support_filter.__file__),
                 ROOT/'research/keyprint_candidate_v3_caller.py']
        for p in files:
            dependencies[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        spec = dict(version=VERSION, inherited_v2_identity=inherited,
                    inherited_engine_identity=inherited['specification']['inherited_engine_identity'],
                    randomness_api='independent_uniform_random_bits(k)_integer_v2',
                    input_contract='raw_float32_model_head_shared_filter_before_every_step',
                    shared_filter=policy, dependencies=dependencies,
                    ordinary_distribution_changed=True, empirical_results_transfer=False)
        self._identity = dict(version=VERSION, runtime_profile_sha256=digest(spec),
            score_namespace_sha256=inherited['score_namespace_sha256'],
            max_steps=max_steps, deployment_calibrated=False, specification=spec)

    @property
    def filter_settings(self):
        return copy.deepcopy(self._filter_settings)

    def pipeline(self, key, *, condition, purpose='general', source_text=None,
                 allow_thinking=False, allow_tools=False):
        self._base._scorer.check_key(key)
        raw = V3Host(key, condition=condition, binding=self._base._binding,
            request=SourceRequest(purpose, source_text),
            channels=ChannelRequest(allow_thinking, allow_tools),
            v2_identity=self.identity, filter_settings=self.filter_settings)
        return Pipeline(raw, hashlib.sha256(key).hexdigest())

    def score_literal(self, text, key):
        report = super().score_literal(text, key)
        report['score_identity_scope'] = 'Inherited score law; v3 changes ordinary filtering and sampling runtime. No empirical v1/v2 calibration transfers.'
        return report


class V3Host(V2Host):
    def __init__(self, *args, filter_settings, **kwargs):
        super().__init__(*args, **kwargs)
        self._filter_settings = copy.deepcopy(filter_settings)
        self.filter_attempts = []

    def step(self, raw_logits, random_bits):
        self._ready()
        before = len(self._all_ids)
        attempt = dict(committed_before=before, committed_after=before,
                       phase='shared_filter', status='started')
        self.filter_attempts.append(attempt)
        try:
            if not isinstance(raw_logits, np.ndarray) or raw_logits.shape != (1,self.model_size):
                raise ValueError('v3 requires the complete raw model head')
            attempt['raw_logits_sha256'] = hashlib.sha256(raw_logits.tobytes()).hexdigest()
            filtered = support_filter.stable_support_filter(raw_logits,
                **self._filter_settings, mapped_vocabulary_size=self.mapped_size)
            attempt.update(filter_profile_sha256=filtered.identity['filter_profile_sha256'],
                           admitted_token_ids=list(filtered.admitted_token_ids),
                           filtered_logits_sha256=hashlib.sha256(filtered.filtered_logits.tobytes()).hexdigest(),
                           diagnostics=filtered.diagnostics, phase='keyed_sampling')
            result = super().step(filtered.filtered_logits, random_bits)
            attempt.update(status='committed', phase='complete')
            return result
        except BaseException as exc:
            attempt.update(status='failed_after_commit' if len(self._all_ids)>before else 'failed_before_commit',
                           exception_type=type(exc).__name__)
            if not self._terminal:
                self._close()
            raise
        finally:
            attempt['committed_after'] = len(self._all_ids)


class Pipeline(V2Pipeline):
    def step(self, raw_logits, random_bits):
        """One raw float32 model head. Filtering is always performed internally."""
        return super().step(raw_logits, random_bits)

    def receipt(self):
        return {**super().receipt(), 'shared_filter_attempts':copy.deepcopy(self._raw.filter_attempts),
                'input_contract':'raw_float32_model_head_shared_filter_before_every_step'}
