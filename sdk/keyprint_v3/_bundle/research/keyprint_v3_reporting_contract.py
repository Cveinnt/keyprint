"""Standalone, unintegrated public-report contract. No scoring or model calls.

Input is an explicit typed projection, not an arbitrary raw receipt. Formatting
does not authenticate caller-supplied identities or validate empirical evidence.
Full API/package/browser coverage is still required for A22/H06 acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from keyprint_reporting_v2 import report_literal_diagnostic

SCHEMA = 'keyprint.public-report.v3-proposed'
VERSION = 'keyprint-candidate-v3-shared-support-2026-09-09'
SAFE_INT = 2**53 - 1
HEX = re.compile(r'[0-9a-f]{64}\Z')
DECIMAL = re.compile(r'(?:0|[1-9][0-9]*)\Z')
NAME = re.compile(r'[a-z][a-z0-9_]{0,79}\Z')
INTERPRETATION = (
    'This report does not identify an author, owner or AI provider. A keyed '
    'diagnostic is uncalibrated; it supplies no detector verdict. Evidence from a '
    'separately calibrated detector could concern involvement in a marked '
    'generation process, including editing or translation, rather than original '
    'authorship. False positives and missed marks are possible. Short or empty '
    'text may provide too little evidence; length alone does not establish power. '
    'A weak, wrong-key, unavailable or failed result does not establish human '
    'authorship. Generation mode is assigned experimental metadata. Verification '
    'checks concern fixture integrity only.'
)


def _keys(value, required, optional=()):
    if type(value) is not dict:
        raise TypeError('typed object must be a dict')
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError('unknown or missing typed fields')


def _json(value):
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int and -SAFE_INT <= value <= SAFE_INT:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value: _json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values(): _json(item)
        return
    raise ValueError('JSON requires finite numbers and exact safe integers; use typed decimal strings for large integers')


def _hash(value):
    if type(value) is not str or HEX.fullmatch(value) is None:
        raise ValueError('invalid SHA256 digest')
    return value


def _count(value, maximum=SAFE_INT):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('invalid bounded nonnegative count')
    return value


def _decimal(value):
    if type(value) is not str or len(value) > 4096 or DECIMAL.fullmatch(value) is None:
        raise ValueError('exact integer must be a canonical bounded decimal string')
    return int(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _target(raw):
    _keys(raw, ('version', 'runtime_profile_sha256', 'score_namespace_sha256',
                'max_steps', 'deployment_calibrated', 'specification'))
    _json(raw)
    if raw['version'] != VERSION or raw['deployment_calibrated'] is not False:
        raise ValueError('uncalibrated proposed v3 target required')
    spec = raw['specification']
    if type(spec) is not dict or _hash(raw['runtime_profile_sha256']) != _digest(spec):
        raise ValueError('target specification digest mismatch')
    if spec.get('version') != VERSION or spec.get('input_contract') != 'raw_float32_model_head_shared_filter_before_every_step':
        raise ValueError('target input contract mismatch')
    engine = spec.get('inherited_engine_identity', {})
    if (type(engine) is not dict or engine.get('deployment_calibrated') is not False
            or spec.get('empirical_results_transfer') is not False
            or _hash(raw['score_namespace_sha256']) != engine.get('score_namespace_sha256')
            or not 1 <= _count(raw['max_steps'], 2048) <= 2048
            or raw['max_steps'] != engine.get('max_steps')):
        raise ValueError('target bound or score namespace mismatch')
    result = {key: copy.deepcopy(raw[key]) for key in raw if key != 'specification'}
    result['specification_digest_verified'] = True
    result['caller_identity_authentication'] = 'not_performed'
    return result, engine


def _scorer(raw, target, engine):
    _keys(raw, ('runtime_profile_sha256', 'score_namespace_sha256', 'runtime_max_steps'))
    if (_hash(raw['runtime_profile_sha256']) != engine.get('runtime_profile_sha256')
            or _hash(raw['score_namespace_sha256']) != target['score_namespace_sha256']
            or raw['runtime_max_steps'] != target['max_steps']):
        raise ValueError('scorer identity does not match target inherited score law')
    _count(raw['runtime_max_steps'], 2048)
    return copy.deepcopy(raw)


def _literal(raw, scorer):
    _keys(raw, ('status', 'runtime_profile_sha256', 'score_namespace_sha256',
        'key_commitment', 'text_sha256', 'runtime_max_steps', 'deployment_calibrated',
        'verdict'), ('events', 'ones', 'trials', 'random_key_null_p', 'underflow_floor', 'trace_sha256'))
    for key, value in scorer.items():
        if raw.get(key) != value: raise ValueError('literal scorer identity mismatch')
    # Existing pure validator preserves numbers; this never calls a scorer.
    return report_literal_diagnostic(raw)['diagnostic']


def _sampling(records, token_ids):
    if type(records) is not list or len(records) != len(token_ids):
        raise ValueError('one completed sampling record per committed token required')
    for step, record in enumerate(records):
        _keys(record, ('step', 'channel', 'token_id', 'base_probability_sha256',
                       'prepared_probability_sha256', 'randomness'))
        if _count(record['step']) != step or _count(record['token_id'], 151668) != token_ids[step]:
            raise ValueError('sampling record and committed tokens disagree')
        if record['channel'] not in ('visible', 'reasoning', 'tool_0', 'tool_1', 'tool_2', 'tool_3'):
            raise ValueError('invalid sampling channel')
        _hash(record['base_probability_sha256']); _hash(record['prepared_probability_sha256'])
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


def _generation(raw, target, scorer):
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
        'sampling_records': _sampling(raw['sampling_records'], tokens),
        'literal_diagnostics': [_literal(d, scorer) for d in diagnostics]}


def _verification(raw, target):
    _keys(raw, ('status', 'artifact_sha256', 'evaluated_runtime_profile_sha256', 'checks'))
    _hash(raw['artifact_sha256'])
    if raw['evaluated_runtime_profile_sha256'] != target['runtime_profile_sha256']:
        raise ValueError('verification target mismatch')
    checks = raw['checks']
    if type(checks) is not list or not checks: raise ValueError('nonempty integrity checks required')
    names = set()
    for check in checks:
        _keys(check, ('name', 'status'))
        if type(check['name']) is not str or not NAME.fullmatch(check['name']) or check['name'] in names:
            raise ValueError('invalid or duplicate integrity check name')
        names.add(check['name'])
        if check['status'] not in ('pass', 'fail'): raise ValueError('invalid integrity check status')
    expected = 'pass' if all(c['status'] == 'pass' for c in checks) else 'fail'
    if raw['status'] != expected: raise ValueError('contradictory overall integrity status')
    return {**copy.deepcopy(raw), 'scope': 'fixture_integrity_only', 'is_detection_result': False}


def _error(raw, target):
    _keys(raw, ('code', 'phase', 'committed_tokens', 'bit_requests',
                'bit_values_obtained', 'bit_values_journaled'))
    if raw['code'] not in ('invalid_input', 'resource_limit', 'interrupted', 'unavailable', 'runtime_failure', 'journal_failure'):
        raise ValueError('unknown error code')
    if type(raw['phase']) is not str or not NAME.fullmatch(raw['phase']): raise ValueError('invalid error phase')
    _count(raw['committed_tokens'], target['max_steps'])
    requested, obtained, journaled = [_count(raw[key]) for key in ('bit_requests', 'bit_values_obtained', 'bit_values_journaled')]
    if not journaled <= obtained <= requested: raise ValueError('contradictory random-consumption counters')
    return {**copy.deepcopy(raw), 'diagnostic_availability': 'unavailable',
            'internal_random_consumption': 'not_inferred_from_returned_values', 'is_negative_detection': False}


def build_report(kind, *, target_identity, payload, scorer_identity=None):
    """Format caller-provided evidence once; no detection/re-evaluation occurs."""
    if kind not in ('literal_diagnostic', 'generation_trace', 'verification', 'error'):
        raise ValueError('unknown report kind')
    _json(payload)
    target, engine = _target(target_identity)
    scorer = _scorer(scorer_identity, target, engine) if scorer_identity is not None else None
    if kind == 'literal_diagnostic':
        if scorer is None: raise ValueError('literal diagnostic requires a distinct scorer identity')
        data = _literal(payload, scorer)
    elif kind == 'generation_trace': data = _generation(payload, target, scorer)
    else:
        if scorer is not None: raise ValueError('non-scoring reports cannot claim a scorer identity')
        data = _verification(payload, target) if kind == 'verification' else _error(payload, target)
    result = {'schema': SCHEMA, 'kind': kind, 'integration_status': 'standalone_unintegrated_component',
        'target_identity': target, 'scorer_identity': scorer, 'payload': data,
        'verdict': None, 'attribution': {'status': 'not_established', 'author': None, 'provider': None},
        'ownership': {'status': 'not_established', 'owner': None},
        'calibration': {'status': 'unavailable', 'deployment_calibrated': False, 'threshold': None},
        'power': {'status': 'not_established', 'evidence_sufficiency': 'insufficient_for_calibrated_conclusion',
                  'minimum_length_guarantee': None},
        'error_rates': {'false_positive_rate': None, 'miss_rate': None,
                        'false_positives_possible': True, 'missed_marks_possible': True},
        'interpretation': INTERPRETATION, 'all_reporting_surfaces_accepted': False}
    _json(result)
    return copy.deepcopy(result)
