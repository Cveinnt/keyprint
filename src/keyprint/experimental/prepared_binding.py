"""Candidate-local unkeyed setup for the explicitly selected native runtime.

The frozen constructor remains the authority at preparation time. Each request
rechecks the pinned files, immutable binding fields and channel markers. Nothing
keyed, sampled or request-owned is cached here.
"""
import copy
from dataclasses import dataclass
import hashlib
from threading import get_ident

import numpy as np

from .._engine.legacy._impl.research import grouped_model_support_boundary as boundary
from .._engine.legacy._impl.research.byte_trie_source_policy import SourceRequest, digest_spec
from .._engine.legacy._impl.research.token_channel_host import (
    Carrier, ChannelRequest, EOS, TOOL_OPEN, TOOL_CLOSE, THINK_OPEN, THINK_CLOSE, route_spec,
)
from .._engine.legacy._impl.research.token_runtime_binding import validate_runtime_binding
from .._engine.legacy._impl.research.token_runtime_profile import RuntimeBoundProfile
from .._engine.legacy._impl.research.token_source_policy import TokenSourceBinding
from .._engine.research.keyprint_candidate_v2.adapter import digest


def _fields(profile):
    return (profile.tokenizer_identity, profile.digest, profile._domain,
            profile.score_namespace_sha256, profile.original_max_steps,
            type(profile.config), profile.config.history, profile.config.layers,
            profile.config.max_steps)


@dataclass(frozen=True, init=False)
class PreparedBinding:
    """Immutable validation snapshot; source/profile replacements fail closed."""
    binding: object
    profile: object
    pieces: tuple
    classes: tuple
    fields: tuple
    sizes: tuple
    excluded_bytes: bytes
    channel_profile: str

    def __init__(self, binding):
        identity = validate_runtime_binding(binding)
        if type(binding.token_bytes) is not tuple or type(binding.profile.classes) is not tuple:
            raise ValueError('prepared binding requires immutable tuple mappings')
        sizes = boundary.declared_sizes()
        profile = binding.profile
        values = dict(binding=binding, profile=profile, pieces=binding.token_bytes,
                      classes=profile.classes, fields=_fields(profile), sizes=sizes,
                      excluded_bytes=np.array([i for i, b in enumerate(binding.token_bytes)
                                               if b is None], dtype=np.int64).tobytes(),
                      channel_profile=digest_spec({'version': 'token-runtime-channel-host-v1',
                                                   'runtime': identity, 'route': route_spec()}))
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def validate(self, binding):
        if (type(binding) is not TokenSourceBinding or binding is not self.binding
                or type(binding.profile) is not RuntimeBoundProfile
                or binding.profile is not self.profile
                or binding.token_bytes is not self.pieces
                or binding.profile.classes is not self.classes
                or _fields(binding.profile) != self.fields):
            raise ValueError('prepared runtime binding changed')
        # Preserve the reference's per-request on-disk integrity check. Parsing
        # and dense-ID reconstruction are reusable only for these exact bytes.
        for path, expected in ((boundary.CONFIG_PATH, boundary.CONFIG_SHA),
                               (boundary.TOKENIZER_PATH, boundary.TOKENIZER_SHA)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('model/tokenizer configuration hash mismatch')
        for token, content in ((EOS, '<|im_end|>'), (TOOL_OPEN, '<tool_call>'),
                               (TOOL_CLOSE, '</tool_call>'), (THINK_OPEN, '<think>'),
                               (THINK_CLOSE, '</think>')):
            if binding._tokenizer.id_to_token(token) != content or binding.token_bytes[token] is not None:
                raise ValueError('pinned channel token mapping differs')

    def initialize(self, host, key, *, condition, binding, request, channels,
                   v2_identity, filter_settings):
        """Same final constructor state as V3Host, with fresh mutable ownership."""
        self.validate(binding)
        if type(request) is not SourceRequest or type(channels) is not ChannelRequest:
            raise TypeError('explicit source and channel requests required')
        host.binding = binding
        host.model_size, host.mapped_size = self.sizes
        host._key, host.condition, host.channels = key, condition, channels
        inherited_route = digest_spec({'channel_profile': self.channel_profile,
                                       'request': request.receipt(), 'channels': channels.__dict__})
        host.channel_profile_sha256 = digest({'v2_runtime': v2_identity['runtime_profile_sha256'],
                                             'inherited_channel_profile': self.channel_profile})
        host.host_route_sha256 = digest({'v2_channel': host.channel_profile_sha256,
                                        'inherited_host_route': inherited_route})
        host.runtime_identity = copy.deepcopy(v2_identity)
        host._owner, host._terminal, host._selected_eos = get_ident(), False, False
        host._all_ids, host._control_ids = [], []
        host._visible = Carrier('visible', binding, key, condition, request)
        host._reasoning, host._tools, host._current = None, [], host._visible
        host.last_sampled_channel = None
        host._excluded = np.frombuffer(self.excluded_bytes, dtype=np.int64)
        host.sampling_records, host.sampling_attempts, host.filter_attempts = [], [], []
        host._filter_settings = copy.deepcopy(filter_settings)
        host._upgrade(host._visible)
