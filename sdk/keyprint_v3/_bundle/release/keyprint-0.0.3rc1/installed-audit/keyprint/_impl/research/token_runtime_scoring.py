"""Independent one-carrier score using only the locked tokenizer and a supplied key.

No primary binding/replay/bit/score imports; no model, study-key file or calibration
corpus loader. These are random-key reference statistics, not detector verdicts.
"""
import hashlib
import hmac
import json
import math
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
NAMESPACE = "690971bdf929e35ecb134916c5ff4b5878562114c802d917face52fc7ee75507"
TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "release/tokenizer/qwen3-8b-4bit/tokenizer.json"
MIN_POSITIVE = float.fromhex("0x0.0000000000001p-1022")


def encode_parts(parts):
    out = len(parts).to_bytes(4, "big")
    for part in parts:
        out += len(part).to_bytes(8, "big") + part
    return out


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reference_tail(ones, trials):
    if type(ones) is not int or type(trials) is not int or not 0 <= ones <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    if ones == 0:
        return 1., False
    # Sum the decreasing side of the PMF relative to its first term. Using the
    # complement below the mean avoids an unstable sum starting at an extreme.
    complement = ones <= trials / 2
    k = ones - 1 if complement else ones
    log_first = math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1) - trials * math.log(2)
    term, terms = 1., [1.]
    if complement:
        while k > 0 and term:
            term *= k / (trials - k + 1)
            k -= 1; terms.append(term)
    else:
        while k < trials and term:
            term *= (trials - k) / (k + 1)
            k += 1; terms.append(term)
    value = math.exp(log_first + math.log(math.fsum(terms)))
    if complement:
        value = 1. - value
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ArithmeticError("invalid independent binomial tail")
    return max(value, MIN_POSITIVE), value == 0.


class RuntimeTextScorer:
    def __init__(self, *, max_steps, tokenizer_path=TOKENIZER_PATH):
        if type(max_steps) is not int or max_steps not in (1024, 2048):
            raise ValueError("runtime bound must be 1024 or 2048")
        raw = Path(tokenizer_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA:
            raise ValueError("locked tokenizer hash mismatch")
        spec = json.loads(raw)
        if spec["model"]["type"] != "BPE" or spec["decoder"]["type"] != "ByteLevel":
            raise ValueError("unsupported tokenizer mapping")
        kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        alphabet = {chr(b): b for b in kept}
        for n, b in enumerate(b for b in range(256) if b not in kept):
            alphabet[chr(256 + n)] = b
        vocab = spec["model"]["vocab"]
        excluded = {row["id"] for row in spec["added_tokens"]}
        ids = set(vocab.values()) | excluded
        if ids != set(range(151669)):
            raise ValueError("locked token ID support differs")
        pieces = [None] * len(ids)
        for symbol, token in vocab.items():
            if token not in excluded:
                pieces[token] = bytes(alphabet[c] for c in symbol)
        self.pieces = tuple(pieces)
        self.tokenizer = Tokenizer.from_str(raw.decode())
        self.max_steps = max_steps
        self.runtime_digest = json_digest({"version": "token-runtime-bound-v1", "score_namespace_sha256": NAMESPACE,
            "original_max_steps": 512, "max_steps": max_steps,
            "context_history": "continuous_no_reset_no_eviction", "keyed_addresses": "unchanged_original_profile_domain"})
        self.domain = encode_parts([b"wm-probe/grouped-canonical/research-v1", bytes.fromhex(NAMESPACE)])

    def identity(self):
        return {"runtime_profile_sha256": self.runtime_digest, "score_namespace_sha256": NAMESPACE,
                "runtime_max_steps": self.max_steps, "deployment_calibrated": False}

    def check_identity(self, receipt):
        if type(receipt) is not dict:
            raise TypeError("runtime scoring identity must be a dict")
        for field in ("runtime_profile_sha256", "score_namespace_sha256"):
            if receipt.get(field) != self.identity()[field]:
                raise ValueError("runtime scoring identity mismatch: " + field)
        for field, value in (("runtime_max_steps", self.max_steps), ("max_steps", self.max_steps),
                             ("original_max_steps", 512), ("score_profile_sha256", NAMESPACE)):
            if field in receipt and (type(receipt[field]) is not type(value) or receipt[field] != value):
                raise ValueError("runtime scoring identity mismatch: " + field)
        if receipt.get("deployment_calibrated", False) is not False:
            raise ValueError("runtime scorer has no deployment calibration")

    @staticmethod
    def check_key(key):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("key must contain exactly 32 bytes")

    def decode_ids(self, ids):
        if len(ids) > self.max_steps:
            raise ValueError("carrier exceeds declared runtime bound")
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self.pieces) or self.pieces[token] is None:
                raise ValueError("control, padding or invalid token in carrier")
        text = b"".join(self.pieces[token] for token in ids).decode("utf-8", "strict")
        if self.tokenizer.decode(list(ids), skip_special_tokens=False) != text:
            raise ValueError("tokenizer decoder differs from literal carrier bytes")
        return text

    def score_ids(self, ids, key, *, identity):
        self.check_key(key); self.check_identity(identity)
        if type(ids) not in (list, tuple):
            raise TypeError("carrier token IDs must be a bounded list or tuple")
        if len(ids) > self.max_steps:
            raise ValueError("carrier exceeds declared runtime bound")
        ids = tuple(ids)
        text = self.decode_ids(ids)
        history, seen, trace, count, ones = (), set(), [], 0, 0
        for token in ids:
            label = self.pieces[token].translate(None, b" \t\r\n\f\v")
            if not label:
                continue
            fresh, bits = history not in seen, None
            if fresh:
                bits = [hmac.new(key, encode_parts([self.domain, encode_parts(history),
                    layer.to_bytes(4, "big"), label]), hashlib.sha256).digest()[0] & 1 for layer in range(30)]
                count += 1; ones += sum(bits)
            trace.append({"context": [p.hex() for p in history], "label": label.hex(), "eligible": fresh, "bits": bits})
            seen.add(history)
            history = (*history, label)[-4:]
        trials = 30 * count
        p, floor = reference_tail(ones, trials) if trials else (None, False)
        return {**self.identity(), "status": "available" if trials else "unavailable",
                "reason": None if trials else "no eligible watermark events",
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "token_ids": ids,
                "events": count, "ones": ones, "trials": trials, "random_key_null_p": p,
                "underflow_floor": floor, "trace_sha256": json_digest(trace),
                "scope": "one carrier, random-key reference statistic; no empirical or authorship verdict"}

    def score_text(self, text, key, *, identity):
        self.check_key(key); self.check_identity(identity)
        if type(text) is not str:
            raise TypeError("carrier text must be str")
        raw = text.encode("utf-8", "strict")
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("carrier text exceeds byte bound")
        try:
            ids = tuple(self.tokenizer.encode(text, add_special_tokens=False).ids)
            if self.decode_ids(ids) != text:
                raise ValueError("tokenizer normalizes literal carrier text")
        except ValueError as exc:
            return {**self.identity(), "status": "unavailable", "reason": str(exc),
                    "text_sha256": hashlib.sha256(raw).hexdigest()}
        return self.score_ids(ids, key, identity=identity)
