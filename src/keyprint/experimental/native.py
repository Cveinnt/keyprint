"""Exact-sampling state for explicitly experimental CPU serving adapters."""
from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
import secrets
import uuid

import numpy as np

from .._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from .._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
from .._engine.research.keyprint_candidate_v3_caller import DurableJournal
from .._engine.research.keyprint_exact_categorical_v2 import sample_float_weights, supported_softmax
from .._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter


class RequestSampler:
    def __init__(self, binding, key, condition, directory, *, runtime="vllm"):
        self.runtime = runtime
        self.binding = binding
        self.profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
        self.session = SparseTokenSourceSession(self.profile, key, condition=condition)
        self.selected = []
        self.journal = DurableJournal(directory / (uuid.uuid4().hex + ".jsonl"))
        self.journal.append({"phase": "start", "condition": condition, "profile_sha256": self.profile.digest,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {p: importlib.metadata.version(p) for p in (runtime, "torch", "numpy", "tokenizers")},
            "scope": "experimental selections; final host commit and quality not certified"})

    def __call__(self, output_ids, logits):
        if list(output_ids) != self.selected:
            self.journal.append({"phase": "host_prefix_mismatch", "expected": self.selected,
                                 "actual": list(output_ids)})
            raise RuntimeError("host prefix differs from previously selected tokens; resampling/speculation unsupported")
        self.journal.append({"phase": "prefix_confirmed", "token_ids": list(output_ids)})
        raw = logits.detach().cpu().numpy()
        if raw.dtype != np.float32 or raw.shape != (len(self.binding.pieces),):
            raise ValueError("CPU float32 head must exactly match the tokenizer binding")
        if np.isnan(raw).any() or np.isposinf(raw).any():
            raise ValueError("invalid model head")
        raw = raw.copy().reshape(1, -1)
        for i, piece in enumerate(self.binding.pieces):
            if piece is None and i not in self.binding.eos_ids:
                raw[0, i] = -np.inf
        filtered = stable_support_filter(raw, temperature=.7, top_k=100,
                                         mapped_vocabulary_size=len(self.binding.pieces))
        base = np.array(supported_softmax(tuple(map(float, filtered.filtered_logits[0]))), dtype=np.float64)
        step = self.session.prepare(base)

        def draw(bits):
            self.journal.append({"phase": "random_requested", "bits": bits})
            value = secrets.randbits(bits)
            self.journal.append({"phase": "random_returned", "bits": bits, "value": value})
            return value

        sample = sample_float_weights(tuple(map(float, step.probabilities)), draw)
        token = sample.token_index
        self.journal.append({"phase": "selected_tentative", "token_id": token, "draw": asdict(sample)})
        self.session.commit(step, token)
        self.selected.append(token)
        logits.fill_(float("-inf"))
        logits[token] = 0.
        return logits

    def close(self):
        self.journal.close()
        self.session.close()
