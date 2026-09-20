"""Experimental CPU GGUF adapter. Keyprint owns sampling; llama.cpp owns logits.

Only decoder-only, non-recurrent GPT-2 byte-BPE vocabularies are admitted.
Native byte APIs avoid the pinned wrapper's fixed-size token decoder and its
unpopulated score cache when logits_all=False. Full-sequence llama_detokenize
may normalize punctuation spacing; this profile preserves raw token pieces.
No Ollama or GPU claim follows.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
from threading import Lock

from .bytelevel import ByteLevelBinding
from .portable import PortableGeneration
from ..sampling import identity as sampling_identity


MODEL_ID = "unsloth/SmolLM2-135M-Instruct-GGUF"
REVISION = "9e6855bc4be717fca1ef21360a1db4b29d5c559a"
MODEL_FILE = "SmolLM2-135M-Instruct-Q8_0.gguf"
MODEL_SHA256 = "c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a"
WRAPPER_VERSION = "0.3.35"


def _native_bytes(call, *, initial=32, limit=1 << 20):
    """Honor a native negative required length, bounded before allocation."""
    buffer = ctypes.create_string_buffer(initial)
    length = call(buffer, len(buffer))
    if length < 0:
        if -length > limit:
            raise ValueError("native byte output exceeds the adapter limit")
        buffer = ctypes.create_string_buffer(-length)
        length = call(buffer, len(buffer))
    if not 0 <= length <= len(buffer):
        raise ValueError("native byte output returned an invalid length")
    return bytes(buffer[:length])


def _binding(model, native):
    meta = model.metadata
    if (meta.get("tokenizer.ggml.model") != "gpt2"
            or meta.get("tokenizer.ggml.add_space_prefix", "false") != "false"
            or native.llama_vocab_type(model._model.vocab) != 2):
        raise ValueError("GGUF requires a GPT-2 byte-BPE vocabulary without a space prefix")
    size = model.n_vocab()
    if not 1 <= size <= 151669:
        raise ValueError("GGUF vocabulary must contain 1 to 151669 tokens")
    pieces, eos = [], set()
    for index in range(size):
        attr = native.llama_token_get_attr(model._model.vocab, index)
        if attr not in (native.LLAMA_TOKEN_ATTR_NORMAL, native.LLAMA_TOKEN_ATTR_CONTROL):
            raise ValueError("GGUF contains an unsupported token attribute")
        stop = native.llama_token_is_eog(model._model.vocab, index)
        if stop:
            if attr != native.LLAMA_TOKEN_ATTR_CONTROL:
                raise ValueError("GGUF end-of-generation tokens must be controls")
            eos.add(index)
        raw = _native_bytes(lambda buf, length: native.llama_token_to_piece(
            model._model.vocab, index, buf, length, 0, False))
        if attr == native.LLAMA_TOKEN_ATTR_CONTROL:
            if raw:
                raise ValueError("GGUF control token unexpectedly emits visible bytes")
            pieces.append(None)
        else:
            if not raw:
                raise ValueError("GGUF ordinary token has no bytes")
            pieces.append(raw)
    if not eos:
        raise ValueError("GGUF must declare an end-of-generation token")
    serialized = json.dumps({"profile": "gguf-byte-bpe-v1-experimental",
        "pieces": [p.hex() if p is not None else None for p in pieces],
        "eos_ids": sorted(eos), "tokenizer_metadata": {
            k: v for k, v in meta.items() if k.startswith("tokenizer.")}},
        sort_keys=True, separators=(",", ":"))
    return ByteLevelBinding(tuple(pieces), frozenset(eos), hashlib.sha256(serialized.encode()).hexdigest())


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LlamaCppModel(PortableGeneration):
    @classmethod
    def load(cls, path: Path, *, temperature=.7, top_k=100, context_size=2048, threads=2):
        if type(context_size) is not int or not 128 <= context_size <= 8192:
            raise ValueError("context_size must be an integer from 128 to 8192")
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer from 1 to 64")
        if not path.is_file():
            raise ValueError("model must be an existing local GGUF file")
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("model must have a GGUF header")
        try:
            from llama_cpp import Llama, llama_cpp as native
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        except ImportError as exc:
            raise ImportError("Install Keyprint with the [llama-cpp] extra") from exc
        if importlib.metadata.version("llama-cpp-python") != WRAPPER_VERSION:
            raise ValueError(f"llama-cpp-python {WRAPPER_VERSION} is required for this native API binding")
        digest = _sha(path)
        if path.name == MODEL_FILE and digest != MODEL_SHA256:
            raise ValueError("the documented GGUF fixture has an unexpected SHA-256")
        model = Llama(model_path=str(path), n_ctx=context_size, n_batch=min(128, context_size),
                      n_threads=threads, n_threads_batch=threads, n_gpu_layers=0,
                      logits_all=False, verbose=False)
        try:
            pointer = model._model.model
            if (native.llama_model_has_encoder(pointer) or native.llama_model_is_recurrent(pointer)
                    or native.llama_model_is_hybrid(pointer)):
                raise ValueError("encoder, recurrent and hybrid GGUF models are unsupported")
            if context_size > model._model.n_ctx_train():
                raise ValueError("context_size exceeds the model's trained context length")
            binding = _binding(model, native)
            template = model.metadata.get("tokenizer.chat_template")
            if not isinstance(template, str) or not template:
                raise ValueError("GGUF requires an explicit chat template")
            def special(index):
                if not 0 <= index < len(binding.pieces):
                    return ""
                return native.llama_token_get_text(model._model.vocab, index).decode("utf-8", errors="strict")
            formatter = Jinja2ChatFormatter(template, eos_token=special(model.token_eos()),
                                           bos_token=special(model.token_bos()))
            instance = cls(model, native, binding, formatter, temperature=temperature, top_k=top_k)
            lib_dir = Path(native._lib._name).resolve().parent
            import llama_cpp
            package = Path(llama_cpp.__file__).parent
            instance.identity.update(model_assets={"gguf_sha256": digest},
                dependencies={name: importlib.metadata.version(name) for name in
                              ("llama-cpp-python", "numpy", "jinja2")},
                wrapper_sources={name: _sha(package / name) for name in
                                 ("llama.py", "llama_cpp.py", "_internals.py", "llama_chat_format.py")},
                native_libraries={p.name: _sha(p) for p in sorted(lib_dir.iterdir())
                                  if p.is_file() and (".so" in p.name or p.suffix in (".dylib", ".dll"))},
                execution={"device": "cpu", "n_gpu_layers": 0, "context_size": context_size,
                           "threads": threads, "batch_size": min(128, context_size), "logits_all": False},
                chat_template_kwargs={"enable_thinking": False})
            return instance
        except BaseException:
            model.close()
            raise

    def __init__(self, model, native, binding, formatter, *, temperature, top_k):
        import numpy as np
        from .._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
        from .._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter
        from ..integrity import verify
        stable_support_filter(np.zeros((1, len(binding.pieces)), dtype=np.float32),
            temperature=temperature, top_k=top_k, mapped_vocabulary_size=len(binding.pieces))
        self.model, self.native, self.binding, self.formatter = model, native, binding, formatter
        self.temperature, self.top_k = temperature, top_k
        self._lock, self._closed = Lock(), False
        self.profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
        self.identity = {"profile": "gguf-byte-bpe-v1-experimental", "profile_sha256": self.profile.digest,
            "tokenizer_binding_sha256": binding.digest, "engine_manifest_sha256": verify()["manifest_sha256"],
            "adapter_sha256": _sha(__file__), "generation_source_sha256": _sha(Path(__file__).with_name("portable.py")),
            "binding_source_sha256": _sha(Path(__file__).with_name("bytelevel.py")),
            "sampling_execution": sampling_identity(), "temperature": temperature, "top_k": top_k,
            "rendering_policy": {"decoder": "native_token_pieces", "cleanup_tokenization_spaces": False,
                                 "utf8": "strict; pending suffix retained only at exact token cap"},
            "empirical_acceptance_transfers": False,
            "limitations": "CPU; single response; no structured output, tools, streaming, batching or calibrated detector"}

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("this llama.cpp model is busy; use one request at a time")
        try:
            if self._closed:
                raise RuntimeError("this llama.cpp model is closed")
            yield
        finally:
            self._lock.release()

    def generate(self, *args, **kwargs):
        with self._exclusive():
            return super().generate(*args, **kwargs)

    def score(self, *args, **kwargs):
        with self._exclusive():
            return super().score(*args, **kwargs)

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close a busy llama.cpp model")
        try:
            if not self._closed:
                self.model.close()
                self._closed = True
        finally:
            self._lock.release()

    @property
    def context_limit(self):
        return self.model.n_ctx()

    def encode_literal(self, text):
        return self.model.tokenize(text.encode("utf-8"), add_bos=False, special=False)

    def encode_prompt(self, prompt):
        rendered = self.formatter(messages=[{"role": "user", "content": prompt}], enable_thinking=False).prompt
        return self.model.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)

    def decode_tokens(self, ids):
        # Re-read native pieces instead of trusting the cached binding table.
        # llama_detokenize is not a verbatim decoder: some BPE families enable
        # clean_spaces internally, deleting French punctuation spaces and
        # changing the byte stream after sampling. Match the Python wrapper's
        # raw-piece policy, with correct resizing for tokens over 32 bytes.
        pieces = [_native_bytes(lambda buf, length: self.native.llama_token_to_piece(
            self.model._model.vocab, index, buf, length, 0, False)) for index in ids]
        return b"".join(pieces).decode("utf-8", errors="strict")

    def json_constraint(self, schema):
        raise ValueError("json_schema requires the MLX or Transformers backend")

    @contextmanager
    def inference_session(self):
        import numpy as np
        self.model.reset()
        def forward(ids):
            self.model.eval(ids)
            pointer = self.model._ctx.get_logits()
            if not pointer:
                raise ValueError("llama.cpp did not return logits")
            # With logits_all=False the wrapper's .scores array is NOT filled.
            # Copy the native final-token head before another eval can overwrite it.
            return np.ctypeslib.as_array(pointer, shape=(len(self.binding.pieces),)).copy()[None, :]
        try:
            yield forward
        finally:
            self.model.reset()
