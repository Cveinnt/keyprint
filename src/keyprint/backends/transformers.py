"""Local Transformers adapter with an explicitly experimental byte-level profile.

Single response, CPU float32, text only. No framework sampler or processors run
after Keyprint selects a token. No retries, batching, tools or streaming.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .bytelevel import ByteLevelBinding
from ..sampling import identity as sampling_identity
from .portable import PortableGeneration


class TransformersModel(PortableGeneration):
    def __init__(self, model: Any, tokenizer: Any, binding: ByteLevelBinding,
                 assets: dict[str, str], *, temperature: float, top_k: int):
        from .._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
        from .._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter
        import numpy as np
        # Validate settings before accepting or loading a prompt.
        stable_support_filter(np.zeros((1, len(binding.pieces)), dtype=np.float32),
                              temperature=temperature, top_k=top_k,
                              mapped_vocabulary_size=len(binding.pieces))
        self.model, self.tokenizer, self.binding = model, tokenizer, binding
        self.temperature, self.top_k = temperature, top_k
        self.profile = Profile(binding.pieces, tokenizer_identity=binding.digest,
                               config=Config(max_steps=1024))
        from ..integrity import verify
        self.identity = {"profile": "portable-bytelevel-v1-experimental", "profile_sha256": self.profile.digest,
                         "tokenizer_binding_sha256": binding.digest, "model_assets": assets,
                         "engine_manifest_sha256": verify()["manifest_sha256"],
                         "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "generation_source_sha256": hashlib.sha256(Path(__file__).with_name("portable.py").read_bytes()).hexdigest(),
                         "binding_source_sha256": hashlib.sha256(Path(__file__).with_name("bytelevel.py").read_bytes()).hexdigest(),
                         "sampling_execution": sampling_identity(),
                         "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
                         "temperature": temperature, "top_k": top_k,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "empirical_acceptance_transfers": False,
                         "limitations": "CPU float32; visible text only; no calibrated detector, tools, reasoning, streaming or batching"}

    @classmethod
    def load(cls, path: Path, *, temperature: float = .7, top_k: int = 100) -> TransformersModel:
        if not path.is_dir():
            raise ValueError("model must be an existing local directory; download an explicit revision first")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install Keyprint with the [transformers] extra") from exc
        config = json.loads((path / "config.json").read_text())
        if config.get("is_encoder_decoder") or config.get("auto_map") or config.get("quantization_config"):
            raise ValueError("encoder-decoder, custom-code and quantized model bindings are unsupported")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if not getattr(tokenizer, "is_fast", False) or not tokenizer.chat_template:
            raise ValueError("a fast tokenizer with an explicit chat template is required")
        eos = config.get("eos_token_id")
        eos = eos if isinstance(eos, list) else [eos]
        binding = ByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),
                vocabulary_size=config["vocab_size"], special_ids=tokenizer.all_special_ids, eos_ids=eos)
        weights = sorted(path.glob("*.safetensors"))
        if not weights:
            raise ValueError("local safetensors weights required")
        assets = {}
        config_files = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "added_tokens.json", "model.safetensors.index.json",
                        "vocab.json", "merges.txt", "chat_template.jinja")
        for file in sorted({*weights, *(path / name for name in config_files if (path / name).is_file())}):
            with file.open("rb") as stream:
                assets[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, dtype=torch.float32).to("cpu").eval()
        return cls(model, tokenizer, binding, assets, temperature=temperature, top_k=top_k)

    @property
    def context_limit(self):
        return self.model.config.max_position_embeddings

    def encode_literal(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_prompt(self, prompt):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False)

    def decode_tokens(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def json_constraint(self, schema):
        from ..structured import JsonConstraint
        return JsonConstraint(schema, self.tokenizer, self.binding)

    @contextmanager
    def inference_session(self):
        import torch
        cache = None
        def forward(ids):
            nonlocal cache
            result = self.model(input_ids=torch.tensor([ids], dtype=torch.long, device="cpu"),
                                past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            return result.logits[:, -1, :].detach().cpu().numpy()
        with torch.inference_mode():
            yield forward
