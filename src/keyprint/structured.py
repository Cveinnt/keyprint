"""Optional JSON grammar applied before filtering and watermark sampling."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from decimal import Decimal

import numpy as np


def _reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _schema_copy(schema: dict) -> dict:
    if type(schema) is not dict:
        raise ValueError("json_schema must be an object")
    try:
        encoded = json.dumps(schema, ensure_ascii=True, allow_nan=False)
        if len(encoded) > 32768:
            raise ValueError("json_schema exceeds 32 KiB")
        copied = json.loads(encoded)
    except (TypeError, RecursionError) as exc:
        raise ValueError("json_schema must contain finite JSON values") from exc
    allowed = {"type", "properties", "required", "additionalProperties", "items",
               "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
               "exclusiveMinimum", "exclusiveMaximum", "enum", "const", "title", "description",
               "anyOf", "allOf", "oneOf"}

    def visit(node, depth=0):
        if depth > 24:
            raise ValueError("json_schema nesting exceeds 24 levels")
        if type(node) is bool:
            return
        if type(node) is not dict or set(node) - allowed:
            raise ValueError("unsupported JSON Schema keyword; references, regex, formats and extensions are not supported")
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("schema properties must be an object")
        for child in properties.values():
            visit(child, depth + 1)
        for name in ("items", "additionalProperties"):
            if name in node:
                visit(node[name], depth + 1)
        for name in ("anyOf", "allOf", "oneOf"):
            if name in node:
                if not isinstance(node[name], list):
                    raise ValueError("schema alternatives must be arrays")
                for child in node[name]:
                    visit(child, depth + 1)

    visit(copied)
    if copied.get("type") != "object":
        raise ValueError("json_schema requires a top-level object")
    return copied


class _Tokenizer:
    def __init__(self, tokenizer, binding):
        self.tokenizer, self.binding = tokenizer, binding
        self.eos_token_id = min(binding.eos_ids)
        self.bos_token_id = None
        self.tokens = [piece if piece is not None else f"<[keyprint-special-{i}]>".encode()
                       for i, piece in enumerate(binding.pieces)]
        self.special_token_ids = [i for i, piece in enumerate(binding.pieces) if piece is None]

    def __call__(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.binding.render(ids) != text:
            raise ValueError("grammar text does not round-trip through the byte binding")
        return ids


class JsonConstraint:
    @classmethod
    def for_mlx(cls, schema, tokenizer):
        """Use exactly the pinned visible-token route, including its sole EOS.

        The MLX head has extra padding entries. They are not grammar tokens;
        the caller masks them rather than inventing a decoder binding.
        """
        from .backends.bytelevel import ByteLevelBinding
        from ._engine.legacy._impl.research.grouped_tokenizer_binding import Binding, TOKENIZER_SHA
        from ._engine.legacy._impl.research.token_channel_host import EOS
        frozen = Binding()
        identity = {"tokenizer_sha256": TOKENIZER_SHA, "eos_ids": [EOS],
                    "pieces": "frozen_visible_route_all_added_tokens_excluded",
                    "vocabulary_size": len(frozen.token_bytes)}
        binding = ByteLevelBinding(frozen.token_bytes, frozenset({EOS}),
            hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
        return cls(schema, tokenizer, binding)

    def __init__(self, schema, tokenizer, binding):
        schema = _schema_copy(schema)
        try:
            from llguidance import LLTokenizer, TokenizerWrapper, LLMatcher, LLParserLimits
            from jsonschema import Draft202012Validator, SchemaError
        except ImportError as exc:
            raise ImportError("Install Keyprint with the [structured] extra") from exc
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError("invalid JSON Schema") from exc
        self.validator = Draft202012Validator(schema)
        grammar = LLMatcher.grammar_from_json_schema(schema, overrides={"lenient": False})
        ll_tokenizer = LLTokenizer(TokenizerWrapper(_Tokenizer(tokenizer, binding)),
                                   eos_token=sorted(binding.eos_ids))
        if ll_tokenizer.vocab_size != len(binding.pieces):
            raise ValueError("grammar vocabulary differs from the model head")
        self.matcher = LLMatcher(ll_tokenizer, grammar, log_level=0,
                                 limits=LLParserLimits(max_lexer_states=10000, max_grammar_size=100000))
        if self.matcher.is_error() or self.matcher.get_grammar_warnings():
            raise ValueError("JSON schema is unsupported or exceeds grammar limits")
        self.size = len(binding.pieces)
        self.identity = {"schema": schema,
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "grammar_sha256": hashlib.sha256(grammar.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("llguidance", "jsonschema")},
            "policy": "grammar_mask_then_base_filter_then_keyprint_sampling",
            "tokenizer_binding_sha256": binding.digest,
            "limits": {"max_lexer_states": 10000, "max_grammar_size": 100000},
            "watermark_detection": "unqualified; constraints may leave no marking capacity"}

    def allowed(self):
        mask = np.frombuffer(self.matcher.compute_logit_bias(), dtype=np.uint8)
        if self.matcher.is_error() or mask.shape != (self.size,) or not np.isin(mask, [0, 200]).all():
            raise ValueError("JSON grammar failed to produce a valid token mask")
        return mask != 0

    def commit(self, token):
        if not self.matcher.consume_token(token) or self.matcher.is_error():
            raise ValueError("sampled token violated the JSON grammar")

    def finish(self, text, completion):
        if completion != "eos":
            return {"status": "incomplete", "schema_validated": False}
        if self.matcher.is_error() or not self.matcher.is_accepting():
            raise ValueError("EOS reached before JSON grammar acceptance")
        # Independent validation never repairs, strips fences or regenerates.
        parsed = json.loads(text, parse_constant=_reject_constant,
                            parse_float=Decimal, object_pairs_hook=_unique_object)
        self.validator.validate(parsed)
        return {"status": "complete", "schema_validated": True}
