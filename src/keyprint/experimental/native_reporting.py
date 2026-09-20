"""Native v2 generation contract: explicit lossless vector commitments.

Sampling and generation invariants are copied from the frozen v3 reporting
contract. Only vector field validation and encoding metadata differ. Legacy
probability hashes are rejected here; legacy reporting remains unchanged.
This formatter validates structure, not unresolved vector content or ownership.
"""
import copy

from .vector_commitment import ENCODING
from .._engine.research.keyprint_v3_reporting_contract import (
    _keys, _count, _hash, _decimal, _literal,
)


def validate_commitment(raw):
    _keys(raw, ('encoding', 'kind', 'width', 'sha256'))
    if raw['encoding'] != ENCODING or raw['kind'] != 'probability':
        raise ValueError('known probability vector commitment required')
    if _count(raw['width'], 151669) != 151669:
        raise ValueError('complete mapped probability width required')
    _hash(raw['sha256'])


def sampling(records, token_ids):
    if type(records) is not list or len(records) != len(token_ids):
        raise ValueError('one completed sampling record per committed token required')
    for step, record in enumerate(records):
        _keys(record, ('step', 'channel', 'token_id', 'base_probability_commitment',
                       'prepared_probability_commitment', 'randomness'))
        if _count(record['step']) != step or _count(record['token_id'], 151668) != token_ids[step]:
            raise ValueError('sampling record and committed tokens disagree')
        if record['channel'] not in ('visible', 'reasoning', 'tool_0', 'tool_1', 'tool_2', 'tool_3'):
            raise ValueError('invalid sampling channel')
        validate_commitment(record['base_probability_commitment'])
        validate_commitment(record['prepared_probability_commitment'])
        raw = record['randomness']
        _keys(raw, ('encoding', 'token_index', 'integer_point_decimal', 'total_weight_decimal', 'transcript'))
        if raw['encoding'] != 'exact-integers-as-decimal-strings-v2': raise ValueError('unknown exact integer encoding')
        _count(raw['token_index'], 151668)
        point, total = _decimal(raw['integer_point_decimal']), _decimal(raw['total_weight_decimal'])
        if not 0 <= point < total: raise ValueError('selected point outside total weight')
        draws = raw['transcript']
        if type(draws) is not list or len(draws) > 1024: raise ValueError('invalid bounded transcript')
        if total == 1:
            if draws or point != 0: raise ValueError('singleton must not consume random bits')
        elif not draws: raise ValueError('non-singleton requires completed random transcript')
        for index, draw in enumerate(draws):
            _keys(draw, ('bit_count', 'value_decimal', 'accepted'))
            count, value = _count(draw['bit_count'], 16384), _decimal(draw['value_decimal'])
            if count != (total - 1).bit_length() or value >= 1 << count:
                raise ValueError('draw outside requested bit domain')
            if type(draw['accepted']) is not bool or draw['accepted'] != (value < total):
                raise ValueError('contradictory random draw acceptance')
            if draw['accepted'] != (index == len(draws) - 1):
                raise ValueError('completed transcript must end at first accepted draw')
        if draws and _decimal(draws[-1]['value_decimal']) != point:
            raise ValueError('selected point differs from accepted draw')
    return copy.deepcopy(records)


def generation(raw, target, scorer):
    _keys(raw, ('condition', 'completion', 'committed_token_ids', 'trace_artifact_sha256',
                'sampling_records', 'literal_diagnostics'))
    if raw['condition'] not in ('ordinary', 'marked') or raw['completion'] not in ('eos', 'length', 'incomplete'):
        raise ValueError('invalid assigned condition or generation completion')
    tokens = raw['committed_token_ids']
    if type(tokens) is not list or len(tokens) > target['max_steps']: raise ValueError('invalid committed token list')
    for token in tokens: _count(token, 151668)
    if (raw['completion'] == 'eos') != (bool(tokens) and tokens[-1] == 151645) or 151645 in tokens[:-1]:
        raise ValueError('completion and committed EOS tokens disagree')
    _hash(raw['trace_artifact_sha256'])
    diagnostics = raw['literal_diagnostics']
    if type(diagnostics) is not list: raise TypeError('literal diagnostics must be a list')
    if diagnostics and scorer is None: raise ValueError('scorer identity required for literal diagnostics')
    return {'assigned_condition': raw['condition'], 'condition_is_inferred_provenance': False,
        'completion': raw['completion'], 'committed_token_ids': copy.deepcopy(tokens),
        'trace_artifact_sha256': raw['trace_artifact_sha256'],
        'artifact_hash_authenticates_content_only_when_resolved': True,
        'sampling_token_index_scope': 'index_into_admitted_positive_support_not_model_token_id',
        'point_to_token_mapping_validation': 'not_performed_requires_resolved_distribution_artifact',
        'sampling_records': sampling(raw['sampling_records'], tokens),
        'vector_commitment_encoding': ENCODING,
        'vector_commitments_resolved': False,
        'literal_diagnostics': [_literal(d, scorer) for d in diagnostics]}

