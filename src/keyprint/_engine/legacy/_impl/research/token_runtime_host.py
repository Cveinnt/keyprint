"""Runtime-bound channel host; supplied model only, explicit dual identities."""
from dataclasses import dataclass, fields

from .byte_trie_source_policy import SourceRequest, digest_spec
from .token_channel_host import ChannelRequest, ChannelStopped, TokenChannelPipeline, route_spec
from .token_runtime_binding import validate_runtime_binding


@dataclass(frozen=True)
class RuntimeChannelStopped(ChannelStopped):
    runtime_profile_sha256: str
    score_namespace_sha256: str


class RuntimeChannelPipeline(TokenChannelPipeline):
    def __init__(self, key, *, condition, binding, request=SourceRequest(), channels=ChannelRequest()):
        identity = validate_runtime_binding(binding)
        super().__init__(key, condition=condition, binding=binding, request=request, channels=channels)
        self.runtime_identity = identity
        self.channel_profile_sha256 = digest_spec({"version": "token-runtime-channel-host-v1",
            "runtime": identity, "route": route_spec()})
        self.host_route_sha256 = digest_spec({"channel_profile": self.channel_profile_sha256,
            "request": request.receipt(), "channels": channels.__dict__})

    def finish(self):
        result = super().finish()
        return RuntimeChannelStopped(**{f.name: getattr(result, f.name) for f in fields(ChannelStopped)},
            runtime_profile_sha256=self.binding.profile.digest,
            score_namespace_sha256=self.binding.profile.score_namespace_sha256)


