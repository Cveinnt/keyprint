"""Pinned Qwen3.5 text-only pilot. Separate profile, no inherited acceptance.

NFC is an input-tokenizer policy, never an output repair. Generated token bytes
are retained exactly. Literal inspection rejects text changed by tokenization.
Only the audited local snapshot and runtime are admitted; no implicit downloads.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from ..backends.bytelevel import ByteLevelBinding
from ..backends.portable import PortableGeneration
from . import wide_sampling

REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"
RUNTIME = {"mlx": "0.32.2", "mlx-lm": "0.31.2", "transformers": "5.16.1"}
# Content pins, rather than a trusted directory name. The baseline independently
# verified these against the Hugging Face snapshot before any generation.
ASSETS = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "README.md": "1407e3e3337063c280b2fe1a63724908a5051cfbf26cd8b98f247432b0382537",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "a96942cb6a8a1d3f1d17514d81a1925d04362a6a3233b389d13012211baaa9f8",
    "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
    "model.safetensors.index.json": "dd023913fb87cfdae27fb11dcf695117c925833796ccac3c64117d6652d8ff1e",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "processor_config.json": "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    "tokenizer_config.json": "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
}


class NFCWideByteLevelBinding(ByteLevelBinding):
    vocabulary_limit = wide_sampling.MAX_VOCABULARY
    allowed_normalizer = {"type": "NFC"}
    profile_name = "keyprint-portable-nfc-wide-bytelevel-v1-experimental"


def verify_assets(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError("model must be an existing local directory; download the pinned revision first")
    # Extra model/config/tokenizer files can influence the upstream loader.
    extras = {p.name for p in path.iterdir() if p.is_file() and
              (p.suffix in {".json", ".safetensors", ".jinja", ".py", ".txt", ".model"})} - set(ASSETS)
    if extras:
        raise ValueError(f"unexpected model assets: {sorted(extras)}")
    for name, expected in ASSETS.items():
        with (path / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"pinned model asset differs: {name}")
    return dict(ASSETS)


class WideMLXModel(PortableGeneration):
    _softmax = staticmethod(wide_sampling.softmax)
    _sample = staticmethod(wide_sampling.sample)

    def _filter(self, raw):
        return wide_sampling.support_filter(raw, temperature=self.temperature, top_k=self.top_k,
                                            mapped_vocabulary_size=len(self.binding.pieces))

    def __init__(self, model, tokenizer, binding, assets, *, temperature=.7, top_k=100):
        from .._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
        from ..integrity import verify
        self.model, self.tokenizer, self.binding = model, tokenizer, binding
        self.temperature, self.top_k = temperature, top_k
        self._filter(np.zeros((1, len(binding.pieces)), np.float32))
        self.profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
        base = Path(__file__).parents[1]
        self.identity = {
            "profile": "portable-nfc-wide-bytelevel-v1-experimental",
            "profile_sha256": self.profile.digest, "tokenizer_binding_sha256": binding.digest,
            "model_id": MODEL_ID, "model_revision": REVISION, "model_assets": assets,
            "engine_manifest_sha256": verify()["manifest_sha256"],
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generation_source_sha256": hashlib.sha256((base / "backends/portable.py").read_bytes()).hexdigest(),
            "binding_source_sha256": hashlib.sha256((base / "backends/bytelevel.py").read_bytes()).hexdigest(),
            "sampling_execution": wide_sampling.identity(),
            "dependencies": {name: importlib.metadata.version(name) for name in (*RUNTIME, "numpy", "tokenizers")},
            "temperature": temperature, "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": False},
            "termination_policy": {"runtime_eos_ids": [248046], "nested_config_eos_id": 248044,
                                   "source": "mlx_lm tokenizer resolution, explicit checked runtime ID"},
            "input_normalizer": {"type": "NFC"}, "output_normalization": "none",
            "empirical_acceptance_transfers": False,
            "limitations": "Experimental local text generation only; no calibrated detector, quality acceptance, tools, JSON grammar, reasoning, streaming or batching",
        }

    @classmethod
    def load(cls, path, *, temperature=.7, top_k=100):
        # Reject settings before expensive hashes or model allocation.
        wide_sampling.support_filter(np.zeros((1, 1), np.float32), temperature=temperature, top_k=top_k)
        assets = verify_assets(path)
        path = Path(path)
        for name, expected in RUNTIME.items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"experimental-wide requires {name}=={expected}")
        from mlx_lm.utils import load_tokenizer
        tokenizer = load_tokenizer(path, {"trust_remote_code": False, "local_files_only": True}, eos_token_ids=None)
        if set(tokenizer.eos_token_ids) != {248046}:
            raise ValueError("Qwen3.5 runtime EOS differs from the pinned policy")
        config = json.loads((path / "config.json").read_text())
        if (config.get("eos_token_id") is not None or config["text_config"]["eos_token_id"] != 248044
                or config["text_config"]["vocab_size"] != 248320):
            raise ValueError("Qwen3.5 model-head or stopping metadata differs")
        binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),
            vocabulary_size=248320, special_ids=tokenizer.all_special_ids, eos_ids=[248046])
        from mlx_lm import load
        model, loaded = load(str(path), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
        if (loaded.backend_tokenizer.to_str() != tokenizer.backend_tokenizer.to_str()
                or set(loaded.eos_token_ids) != {248046}):
            raise ValueError("loaded tokenizer differs from validated binding")
        return cls(model, loaded, binding, assets, temperature=temperature, top_k=top_k)

    @property
    def context_limit(self):
        return 262144  # shared caller enforces the tighter 8192-token budget

    def encode_literal(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_prompt(self, prompt):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def decode_tokens(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @contextmanager
    def inference_session(self):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        def forward(ids):
            logits = self.model(mx.array([ids]), cache=cache)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            return np.array(logits)
        try:
            yield forward
        finally:
            cache.clear()

    def close(self):
        self.model = None
        self.tokenizer = None
