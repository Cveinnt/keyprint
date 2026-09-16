"""Token-ID channel routing on the frozen grouped scoring component."""
import codecs
from dataclasses import dataclass
import hashlib
import json
import math
from threading import get_ident

import numpy as np

from .byte_trie_source_policy import SourceRequest, digest_spec
from .grouped_model_support_boundary import declared_sizes, project_supported_logits
from .grouped_null_score import score
from .token_source_host import TokenFinished
from .token_source_policy import TokenSourceBinding, TokenSourceSession

EOS, TOOL_OPEN, TOOL_CLOSE, THINK_OPEN, THINK_CLOSE = 151645, 151657, 151658, 151667, 151668


@dataclass(frozen=True)
class ChannelRequest:
    allow_thinking: bool = False
    allow_tools: bool = False

    def __post_init__(self):
        if type(self.allow_thinking) is not bool or type(self.allow_tools) is not bool:
            raise TypeError("channel enablement requires booleans")


def route_spec():
    return {"version": "qwen-token-channel-route-v1", "eos": EOS,
        "reasoning_controls": [THINK_OPEN, THINK_CLOSE], "tool_controls": [TOOL_OPEN, TOOL_CLOSE],
        "initial_channel": "visible", "reasoning": "optional_once_before_visible_payload_or_tools",
        "tool_blocks_per_message_cap": 4, "carriers": "persistent_visible_fresh_reasoning_each_tool",
        "marking": "same_token_score_profile_all_generated_carriers",
        "source_policy": "request_on_visible_general_on_other_carriers",
        "control_mass": "preserve_exact_commit_before_transition_or_error",
        "input_tool_responses": "prompt_only_never_generated_carrier_history",
        "incomplete_eos": "error", "incomplete_length": "retain_open_status_no_synthetic_close",
        "carrier_score_aggregation": "forbidden_reused_keyed_addresses"}


class ChannelProtocolError(ValueError):
    pass


class Carrier:
    def __init__(self, name, binding, key, condition, request):
        self.name, self.binding, self.key = name, binding, key
        self.session = TokenSourceSession(binding.profile, key, condition=condition, request=request)
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.ids, self.events, self.text = [], [], ""
        self.closed, self.finished = False, None

    def boundary(self):
        if self.decoder.getstate()[0]:
            raise UnicodeDecodeError("utf-8", self.decoder.getstate()[0], 0,
                len(self.decoder.getstate()[0]), "channel boundary splits a UTF-8 character")

    def append(self, token, event):
        self.ids.append(token)
        if event is not None:
            self.events.append(event)
        emitted = self.decoder.decode(self.binding.token_bytes[token], final=False)
        self.text += emitted
        return emitted

    def finish(self):
        if self.finished is not None:
            return self.finished
        self.text += self.decoder.decode(b"", final=True)
        if self.binding.decode_visible(self.ids) != self.text:
            raise ValueError("carrier sampled bytes and decoded text differ")
        generation = score(self.events, self.binding.profile.config.layers)
        same, unavailable, text_score, note = False, None, None, None
        try:
            replay = self.binding.replay_text(self.text, self.key)
        except ValueError as exc:
            unavailable = str(exc)
        else:
            same = tuple(self.events) == replay.events
            text_score = score(replay.events, self.binding.profile.config.layers)
            if not same:
                note = "text retokenization changes watermark events; both statistics retained"
        self.finished = TokenFinished(self.text, tuple(self.ids), self.binding.profile.digest,
            same, unavailable, generation, text_score, note)
        return self.finished

    def close(self):
        if not self.closed:
            self.session.close()
            self.closed = True


@dataclass(frozen=True)
class ChannelStopped:
    visible: TokenFinished
    reasoning: TokenFinished | None
    tools: tuple[TokenFinished, ...]
    reason: str
    eos_token_id: int | None
    committed_token_ids: tuple[int, ...]
    host_route_sha256: str
    channel_profile_sha256: str
    protocol_complete: bool
    channel_at_stop: str


@dataclass(frozen=True)
class ChannelStep:
    token_id: int
    emitted_text: str
    stopped: ChannelStopped | None


class TokenChannelPipeline:
    def __init__(self, key, *, condition, binding=None, request=SourceRequest(), channels=ChannelRequest()):
        self.binding = binding if binding is not None else TokenSourceBinding()
        if type(self.binding) is not TokenSourceBinding:
            raise ValueError("requires TokenSourceBinding")
        if type(request) is not SourceRequest or type(channels) is not ChannelRequest:
            raise TypeError("explicit source and channel requests required")
        self.model_size, self.mapped_size = declared_sizes()
        self._key, self.condition, self.channels = key, condition, channels
        # Binding verifies tokenizer bytes; verify these particular declared markers too.
        for token, content in ((EOS, "<|im_end|>"), (TOOL_OPEN, "<tool_call>"), (TOOL_CLOSE, "</tool_call>"),
                               (THINK_OPEN, "<think>"), (THINK_CLOSE, "</think>")):
            if self.binding._tokenizer.id_to_token(token) != content or self.binding.token_bytes[token] is not None:
                raise ValueError("pinned channel token mapping differs")
        self.channel_profile_sha256 = digest_spec({"score_profile": self.binding.profile.digest,
            "route": route_spec(), "global_sample_cap": self.binding.profile.config.max_steps})
        self.host_route_sha256 = digest_spec({"channel_profile": self.channel_profile_sha256,
            "request": request.receipt(), "channels": channels.__dict__})
        self._owner, self._terminal, self._selected_eos = get_ident(), False, False
        self._all_ids, self._control_ids = [], []
        self._visible = Carrier("visible", self.binding, key, condition, request)
        self._reasoning, self._tools, self._current = None, [], self._visible
        self.last_sampled_channel = None
        self._excluded = np.array([i for i, b in enumerate(self.binding.token_bytes) if b is None], dtype=np.int64)

    @property
    def committed_token_ids(self): return tuple(self._all_ids)
    @property
    def token_ids(self): return tuple(self._visible.ids)
    @property
    def text_so_far(self): return self._visible.text
    @property
    def carriers(self):
        return [self._visible] + ([self._reasoning] if self._reasoning is not None else []) + list(self._tools)

    def _ready(self):
        if get_ident() != self._owner:
            raise RuntimeError("pipeline belongs to another thread")
        if self._terminal:
            raise RuntimeError("response is terminal; no retry or reset")
        self._current.session._ready()

    def _close(self):
        self._terminal = True
        for carrier in self.carriers:
            carrier.close()

    def _route(self, token):
        current = self._current
        current.boundary()
        if token == THINK_OPEN:
            if (not self.channels.allow_thinking or current is not self._visible or self._reasoning is not None
                    or self._tools or self._visible.session._source_output):
                raise ChannelProtocolError("reasoning open is not permitted in this state")
            self._reasoning = Carrier("reasoning", self.binding, self._key, self.condition, SourceRequest())
            self._current = self._reasoning
        elif token == THINK_CLOSE:
            if current is not self._reasoning or current is None:
                raise ChannelProtocolError("reasoning close without open reasoning")
            current.finish(); current.close(); self._current = self._visible
        elif token == TOOL_OPEN:
            if not self.channels.allow_tools or current is not self._visible or len(self._tools) >= 4:
                raise ChannelProtocolError("tool open is not permitted in this state")
            tool = Carrier(f"tool_{len(self._tools)}", self.binding, self._key, self.condition, SourceRequest())
            self._tools.append(tool); self._current = tool
        elif token == TOOL_CLOSE:
            if current not in self._tools:
                raise ChannelProtocolError("tool close without open tool")
            current.finish(); current.close(); self._current = self._visible
        elif token == EOS:
            if current is not self._visible:
                raise ChannelProtocolError("EOS before generated channel closes")
            self._selected_eos = True
            return self.finish()
        else:
            raise ChannelProtocolError(f"selected unsupported control token {token}")
        return None

    def step(self, filtered_logits, uniform):
        self._ready()
        try:
            if len(self._all_ids) >= self.binding.profile.config.max_steps:
                raise RuntimeError("global assistant sample cap reached")
            if type(uniform) is not float or not math.isfinite(uniform) or not 0 <= uniform < 1:
                raise ValueError("sampling uniform must be a finite float in [0, 1)")
            head = project_supported_logits(filtered_logits, model_size=self.model_size, mapped_size=self.mapped_size)[0]
            weights = np.exp(head.astype(np.float64) - float(np.max(head)))
            base = weights / float(weights.sum())
            current = self._current
            prepared = current.session.prepare(base)
            if not np.array_equal(prepared.probabilities[self._excluded], base[self._excluded]):
                raise ArithmeticError("excluded control probability changed")
            support = np.flatnonzero(prepared.probabilities > 0)
            cdf = np.cumsum(prepared.probabilities[support]); cdf /= cdf[-1]; cdf[-1] = 1.
            token = int(support[np.searchsorted(cdf, uniform, side="right")])
            event = current.session.commit(prepared, token)
            self._all_ids.append(token); self.last_sampled_channel = current.name
            if self.binding.token_bytes[token] is None:
                self._control_ids.append(token)
                if event is not None:
                    raise ArithmeticError("control produced watermark event")
                return ChannelStep(token, "", self._route(token))
            emitted = current.append(token, event)
            return ChannelStep(token, emitted if current is self._visible else "", None)
        except Exception:
            if not self._terminal:
                self._close()
            raise

    def finish(self):
        self._ready()
        try:
            values = {c.name: c.finish() for c in self.carriers}
            return ChannelStopped(values["visible"], values.get("reasoning"),
                tuple(values[t.name] for t in self._tools),
                "declared_eos" if self._selected_eos else "external_stop", EOS if self._selected_eos else None,
                self.committed_token_ids, self.host_route_sha256, self.channel_profile_sha256,
                self._current is self._visible, self._current.name)
        finally:
            self._close()


