"""Explicit optional native PRF execution, with independently bound identity."""
import hashlib
from pathlib import Path
import numpy as np

from .fast_mlx import (FastCandidate, _ContextProfile, _SparseV3Host, _SparseV2Host, _upgrade,
                       execution_specification as python_execution)
from . import partition_filter
from . import prepared_binding
from .fast_public import FastPublicCandidate
from .batched_tournament import BatchedTokenSourceSession
from .capped_utf8 import CappedPipeline
from .._engine.legacy._impl.research.grouped_canonical_prototype import pack
from .._engine.legacy._impl.research.byte_trie_source_policy import SourceRequest
from .._engine.legacy._impl.research.token_channel_host import ChannelRequest
from .._engine.research.keyprint_candidate_v3.adapter import digest

VERSION = 'keyprint-mlx-native-experimental-v1'


def native_backend():
    try:
        from keyprint_native import NativePRF
    except ImportError as exc:
        raise ImportError('experimental-native requires the separate keyprint-native wheel; see native/README.md. No automatic build or download was started.') from exc
    native = NativePRF()
    expected = {'encoding':'little-endian-binary32',
                'order':'descending score, ascending original index; signed zeros tie',
                'maximum_scores':151669,'maximum_top_k':512}
    if (native.identity.get('package_version') != '0.1.0a2'
            or native.identity.get('selection') != expected
            or not callable(getattr(native, 'select_indices', None))):
        raise ImportError('experimental-native requires keyprint-native==0.1.0a2 with bounded selection support; install the matching reviewed wheel before loading a model')
    return native


def execution_specification(native=None):
    result=python_execution()
    result['native_sdk_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['support_filter_execution']={
        'implementation':'bounded-native-top-k-with-exact-partition-fallback-v1',
        'source_sha256':hashlib.sha256(Path(partition_filter.__file__).read_bytes()).hexdigest(),
        'policy':'unchanged reference support law; exact cutoff ties by ascending token ID',
    }
    result['native_prf']=(native or native_backend()).identity
    result['binding_preparation']={
        'implementation':'candidate-local-unkeyed-snapshot-v1',
        'source_sha256':hashlib.sha256(Path(prepared_binding.__file__).read_bytes()).hexdigest(),
        'revalidation':'pinned files, immutable binding fields and channel markers on every request',
    }
    result['request_local_hmac_context']=False
    result['native_batch_key_state']='per-call only; no persistent keyed state'
    return result


class _NativeProfile(_ContextProfile):
    def __init__(self, reference, native):
        super().__init__(reference)
        object.__setattr__(self,'_native',native)

    def table(self,key,context,labels):
        labels=list(labels)
        if type(key) is not bytes or len(key)!=32:
            raise ValueError('key must contain exactly 32 bytes')
        if any(type(label) is not bytes or not label for label in labels):
            raise ValueError('nonempty byte labels required')
        context_bytes=pack(context)
        prefix=(int(4).to_bytes(4,'big')+len(self._domain).to_bytes(8,'big')+self._domain
                +len(context_bytes).to_bytes(8,'big')+context_bytes+int(4).to_bytes(8,'big'))
        result={}
        layers=self.config.layers
        # Bounded batches preserve original label order and handle larger top-k.
        for start in range(0,len(labels),1000):
            batch=labels[start:start+1000]
            suffixes=[len(label).to_bytes(8,'big')+label for label in batch]
            raw=self._native.digests(key,prefix,suffixes,layers)
            for index,label in enumerate(batch):
                result[label]=tuple(raw[(index*layers+layer)*32]&1 for layer in range(layers))
        return result

    def bits(self,key,context,label):
        return self.table(key,context,[label])[label]


class _NativeSession(BatchedTokenSourceSession):
    def __init__(self,profile,key,*,native,**settings):
        super().__init__(_NativeProfile(profile,native),key,**settings)

    @staticmethod
    def _bit_table(profile,key,context,labels):
        return profile.table(key,context,labels)

    def close(self):
        super().close()
        self.profile._state.clear()


class _NativeHost(_SparseV3Host):
    def __init__(self,*args,native,prepared=None,**settings):
        self._native=native
        if prepared is None:
            super().__init__(*args,**settings)
        else:
            if type(prepared) is not prepared_binding.PreparedBinding:
                raise TypeError('exact PreparedBinding required')
            prepared.initialize(self,*args,**settings)

    def _upgrade(self,carrier):
        _upgrade(carrier,_NativeSession,native=self._native)

    def step(self, raw_logits, random_bits):
        self._ready()
        before = len(self._all_ids)
        attempt = dict(committed_before=before, committed_after=before,
                       phase='shared_filter', status='started')
        self.filter_attempts.append(attempt)
        try:
            if not isinstance(raw_logits, np.ndarray) or raw_logits.shape != (1, self.model_size):
                raise ValueError('v3 requires the complete raw model head')
            attempt['raw_logits_sha256'] = hashlib.sha256(raw_logits.tobytes()).hexdigest()
            filtered = partition_filter.partition_support_filter(raw_logits,
                **self._filter_settings, mapped_vocabulary_size=self.mapped_size, native=self._native)
            attempt.update(filter_profile_sha256=filtered.identity['filter_profile_sha256'],
                           admitted_token_ids=list(filtered.admitted_token_ids),
                           filtered_logits_sha256=hashlib.sha256(filtered.filtered_logits.tobytes()).hexdigest(),
                           diagnostics=filtered.diagnostics, phase='keyed_sampling')
            result = _SparseV2Host.step(self, filtered.filtered_logits, random_bits)
            attempt.update(status='committed', phase='complete')
            return result
        except BaseException as exc:
            attempt.update(status='failed_after_commit' if len(self._all_ids) > before else 'failed_before_commit',
                           exception_type=type(exc).__name__)
            if not self._terminal:
                self._close()
            raise
        finally:
            attempt['committed_after'] = len(self._all_ids)


class NativeCandidate(FastCandidate):
    def __init__(self,reference):
        super().__init__(reference)
        self._native=native_backend()
        self._prepared=prepared_binding.PreparedBinding(reference._base._binding)
        spec=self._identity['specification']
        spec['version']=VERSION
        spec['execution']=execution_specification(self._native)
        self._identity.update(version=VERSION,runtime_profile_sha256=digest(spec))

    def pipeline(self,key,*,condition,purpose='general',source_text=None,
                 allow_thinking=False,allow_tools=False):
        self.reference._base._scorer.check_key(key)
        raw=_NativeHost(key,condition=condition,binding=self.reference._base._binding,
                        request=SourceRequest(purpose,source_text),
                        channels=ChannelRequest(allow_thinking,allow_tools),
                        v2_identity=self.identity,filter_settings=self.filter_settings,native=self._native,
                        prepared=self._prepared)
        return CappedPipeline(raw,hashlib.sha256(key).hexdigest())


class NativePublicCandidate(FastPublicCandidate):
    core_type=NativeCandidate
    facade_version='keyprint-experimental-native-facade-v1'
    integration_status='experimental_native_execution_lifecycle_qualification_pending'
    package_scope='Optional native PRF execution; no scientific or serving acceptance transfer'
