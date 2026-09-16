"""CPU-only vLLM 0.29 pilot. Actual request completion needs external verification.

The adapter samples represented weights itself and returns a one-token mask.
Its receipts record selections, not claims that the serving engine committed
the final token. It is deliberately unavailable through the normal CLI.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from ..backends.bytelevel import ByteLevelBinding
from ..cli import load_key
from ..integrity import verify
from .native import RequestSampler


def validate(params):
    expected = {"n": 1, "temperature": 1., "top_p": 1., "min_p": 0.,
                "presence_penalty": 0., "frequency_penalty": 0., "repetition_penalty": 1.,
                "seed": None, "logprobs": None, "prompt_logprobs": None, "min_tokens": 0,
                "ignore_eos": False, "structured_outputs": None, "thinking_token_budget": None}
    for name, value in expected.items():
        if getattr(params, name, None) != value:
            raise ValueError(f"Keyprint pilot requires {name}={value!r}")
    for name in ("stop", "stop_token_ids", "bad_words", "allowed_token_ids", "logit_bias"):
        if getattr(params, name, None):
            raise ValueError(f"Keyprint pilot does not support {name}")
    if params.top_k not in (-1, 0) or not 1 <= params.max_tokens <= 1024:
        raise ValueError("disable framework top-k and request 1 to 1024 tokens")
    extra = params.extra_args or {}
    if set(extra) - {"keyprint_condition"} or extra.get("keyprint_condition", "marked") not in ("ordinary", "marked"):
        raise ValueError("only keyprint_condition=ordinary|marked is supported")


class KeyprintLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        validate(params)

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        if device.type != "cpu" or importlib.metadata.version("vllm") != "0.29.0+cpu":
            raise ValueError("pilot binds only vLLM 0.29.0+cpu")
        if vllm_config.speculative_config is not None:
            raise ValueError("speculative decoding is unsupported")
        verify()
        from transformers import AutoTokenizer
        path = Path(vllm_config.model_config.model)
        if not path.is_dir():
            raise ValueError("local model assets required")
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

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        validate(params)
        return RequestSampler(self.binding, self.key, (params.extra_args or {}).get("keyprint_condition", "marked"), self.directory)

    def update_state(self, batch_update):
        if batch_update:
            # Finished/replaced request samplers are closed before the upstream
            # adapter applies removals, additions and moves.
            removed = set(batch_update.removed) | {item[0] for item in batch_update.added}
            for index in removed:
                if index in self.req_info:
                    self.req_info[index].func.close()
        super().update_state(batch_update)
