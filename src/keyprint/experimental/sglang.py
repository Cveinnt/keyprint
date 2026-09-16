"""Trusted offline SGLang ARM CPU pilot; never expose serialized hooks publicly."""
import importlib.metadata
import json
import os
from pathlib import Path
import weakref

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.sampling_params import TOP_K_ALL

from ..backends.bytelevel import ByteLevelBinding
from ..cli import load_key
from ..integrity import verify
from .native import RequestSampler

_LIVE_REQUESTS = weakref.WeakKeyDictionary()


def validate(params):
    expected = {"temperature": 1., "top_p": 1., "min_p": 0., "n": 1,
                "frequency_penalty": 0., "presence_penalty": 0., "repetition_penalty": 1.,
                "min_new_tokens": 0, "ignore_eos": False, "sampling_seed": None,
                "beam_width": None}
    for name, value in expected.items():
        if getattr(params, name, None) != value:
            raise ValueError(f"Keyprint pilot requires {name}={value!r}")
    for name in ("stop", "stop_strs", "stop_regex", "stop_regex_strs", "stop_token_ids",
                 "json_schema", "regex", "ebnf", "structural_tag", "logit_bias"):
        if getattr(params, name, None):
            raise ValueError(f"Keyprint pilot does not support {name}")
    if params.top_k not in (-1, TOP_K_ALL) or not 1 <= params.max_new_tokens <= 1024:
        raise ValueError("disable framework top-k and request 1 to 1024 tokens")
    extra = params.custom_params or {}
    if set(extra) - {"keyprint_condition", "__req__"} or extra.get("keyprint_condition", "marked") not in ("ordinary", "marked"):
        raise ValueError("unsupported custom parameters")


class KeyprintLogitsProcessor(CustomLogitProcessor):
    def __init__(self):
        if importlib.metadata.version("sglang-cpu") != "0.5.20.dev791+g13d593b6c":
            raise ValueError("pilot requires pinned SGLang CPU source build")
        verify()
        from sglang.srt.runtime_context import get_server_args
        args = get_server_args()
        if args.device != "cpu" or not args.disable_overlap_schedule or args.speculative_algorithm is not None:
            raise ValueError("pilot requires CPU, disabled overlap and no speculation")
        if not args.disable_radix_cache:
            raise ValueError("prefix cache reuse is not validated")
        from transformers import AutoTokenizer
        path = Path(os.environ["KEYPRINT_MODEL_PATH"])
        if str(path) != args.model_path or not path.is_dir():
            raise ValueError("model binding must match the engine's local model assets")
        config = json.loads((path / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        eos = config["eos_token_id"]
        self.binding = ByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),
            vocabulary_size=config["vocab_size"], special_ids=tokenizer.all_special_ids,
            eos_ids=eos if isinstance(eos, list) else [eos])
        self.key = load_key(Path(os.environ["KEYPRINT_KEY_FILE"]))
        self.directory = Path(os.environ["KEYPRINT_TRACE_DIR"])
        if self.directory.is_symlink():
            raise ValueError("trace directory must not be a symlink")
        self.directory.mkdir(mode=0o700, exist_ok=True)
        if self.directory.stat().st_mode & 0o077:
            raise ValueError("trace directory must be private")

    def __call__(self, logits, custom_param_list=None):
        if logits.device.type != "cpu" or custom_param_list is None or len(custom_param_list) != logits.shape[0]:
            raise ValueError("CPU logits and explicit per-row request state required")
        # State follows the request, because SGLang recreates processors when
        # rebuilding batches. At most 32 live requests are retained. There is no
        # claim of production lifecycle/cancellation support or unbounded service.
        for index, extra in enumerate(custom_param_list):
            req = extra.get("__req__")
            if req is None or req.return_logprob:
                raise ValueError("explicit request without logprobs required")
            validate(req.sampling_params)
            condition = extra.get("keyprint_condition", "marked")
            if req not in _LIVE_REQUESTS:
                if len(_LIVE_REQUESTS) >= 32:
                    raise ValueError("experimental session request limit reached")
                sampler = RequestSampler(self.binding, self.key, condition,
                                         self.directory, runtime="sglang-cpu")
                _LIVE_REQUESTS[req] = (self.binding.digest, self.key, condition, sampler)
                weakref.finalize(req, sampler.close)
            digest, key, original_condition, sampler = _LIVE_REQUESTS[req]
            if (digest, key, original_condition) != (self.binding.digest, self.key, condition):
                raise ValueError("request binding, key or condition changed")
            logits[index] = sampler(req.output_ids, logits[index])
        return logits
