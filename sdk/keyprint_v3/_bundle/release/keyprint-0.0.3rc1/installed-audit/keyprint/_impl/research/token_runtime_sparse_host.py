"""Opt-in exact execution engine for the current runtime/channel host.

The reference and all frozen studies stay unchanged. New receipts distinguish
execution code identity from the unchanged channel/runtime/scoring identities.
"""
import hashlib
from pathlib import Path

from .byte_trie_source_policy import SourceRequest, digest_spec
from .token_channel_host import ChannelRequest
from .token_runtime_host import RuntimeChannelPipeline
from .token_source_policy import TokenSourceSession
from .token_source_sparse_execution import SparseTokenSourceSession
from . import token_source_sparse_execution

ENGINE_ID = digest_spec({"version": "token-runtime-sparse-execution-v1",
    "host_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "session_sha256": hashlib.sha256(Path(token_source_sparse_execution.__file__).read_bytes()).hexdigest()})


def upgrade_fresh_carrier(carrier):
    old = carrier.session
    if type(old) is SparseTokenSourceSession:
        return
    if (type(old) is not TokenSourceSession or old._steps or old._pending is not None
            or old._context or old._used or old._source_output or old._closed):
        raise RuntimeError("only an untouched reference carrier may select the sparse engine")
    replacement = SparseTokenSourceSession(old.profile, old._key, condition=old._condition, request=old._request)
    old.close()
    carrier.session = replacement


class SparseRuntimeChannelPipeline(RuntimeChannelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        upgrade_fresh_carrier(self._visible)

    def _route(self, token):
        result = super()._route(token)
        if not self._terminal:
            upgrade_fresh_carrier(self._current)
        return result


