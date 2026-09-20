"""Explicit optional native PRF execution, with independently bound identity."""
import hashlib
from pathlib import Path

from .fast_mlx import (FastCandidate, _ContextProfile, _SparseV3Host, _upgrade,
                       execution_specification as python_execution)
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
    return NativePRF()


def execution_specification(native=None):
    result=python_execution()
    result['native_sdk_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['native_prf']=(native or native_backend()).identity
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
    def __init__(self,*args,native,**settings):
        self._native=native
        super().__init__(*args,**settings)

    def _upgrade(self,carrier):
        _upgrade(carrier,_NativeSession,native=self._native)


class NativeCandidate(FastCandidate):
    def __init__(self,reference):
        super().__init__(reference)
        self._native=native_backend()
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
                        v2_identity=self.identity,filter_settings=self.filter_settings,native=self._native)
        return CappedPipeline(raw,hashlib.sha256(key).hexdigest())


class NativePublicCandidate(FastPublicCandidate):
    core_type=NativeCandidate
    facade_version='keyprint-experimental-native-facade-v1'
    integration_status='experimental_native_execution_lifecycle_qualification_pending'
    package_scope='Optional native PRF execution; no scientific or serving acceptance transfer'
