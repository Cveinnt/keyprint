"""Current local candidate API. No Anthropic key or calibrated authorship verdict."""
from dataclasses import asdict
import hashlib
from threading import get_ident

from ._impl.research.byte_trie_source_policy import SourceRequest
from ._impl.research.token_channel_host import ChannelRequest
from ._impl.research.token_runtime_binding import runtime_binding, validate_runtime_binding
from ._impl.research.token_runtime_scoring import RuntimeTextScorer
from ._impl.research.token_runtime_sparse_host import SparseRuntimeChannelPipeline, ENGINE_ID


class Candidate:
    """Pinned tokenizer/profile; no key, model or file discovery is performed."""
    def __init__(self, *, max_steps=2048):
        self._binding=runtime_binding(max_steps=max_steps)
        self._scorer=RuntimeTextScorer(max_steps=max_steps)
        self.identity=validate_runtime_binding(self._binding)
        self._scorer.check_identity(self.identity)

    def pipeline(self, key, *, condition, purpose='general', source_text=None,
                 allow_thinking=False, allow_tools=False):
        self._scorer.check_key(key)
        raw=SparseRuntimeChannelPipeline(key,condition=condition,binding=self._binding,
            request=SourceRequest(purpose,source_text),channels=ChannelRequest(allow_thinking,allow_tools))
        return Pipeline(raw,hashlib.sha256(key).hexdigest())

    def score_literal(self, text, key):
        """Score the supplied whole carrier; no empirical threshold or truncation."""
        result=self._scorer.score_text(text,key,identity=self.identity)
        return {**result,'key_commitment':hashlib.sha256(key).hexdigest(),
            'verdict':None,'scope':'literal carrier random-key reference score; not calibrated authorship detection'}


class Pipeline:
    def __init__(self,raw,key_commitment):
        self._raw,self._key_commitment=raw,key_commitment
        self._result=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    @property
    def model_vocabulary_size(self):return self._raw.model_size

    @property
    def committed_token_ids(self):return tuple(self._raw._all_ids)

    def step(self,filtered_logits,uniform):
        """One ordinary-filtered NumPy logits row and one explicit uniform draw."""
        value=self._raw.step(filtered_logits,uniform)
        if value.stopped is not None:self._result=value.stopped
        return value

    def finish(self):
        if get_ident()!=self._raw._owner:raise RuntimeError('pipeline belongs to another thread')
        if self._result is None:self._result=self._raw.finish()
        return self._result

    def close(self):
        if get_ident()!=self._raw._owner:raise RuntimeError('pipeline belongs to another thread')
        self._raw._close()

    def receipt(self):
        if get_ident()!=self._raw._owner:raise RuntimeError('pipeline belongs to another thread')
        return dict(runtime=self._raw.runtime_identity,execution_engine_sha256=ENGINE_ID,
            key_commitment=self._key_commitment,condition=self._raw.condition,
            committed_token_ids=list(self.committed_token_ids),
            channel_profile_sha256=self._raw.channel_profile_sha256,host_route_sha256=self._raw.host_route_sha256,
            closed=self._raw._terminal,finalized=self._result is not None,
            final=asdict(self._result) if self._result else None,calibrated=False)


__all__=['Candidate','Pipeline']
