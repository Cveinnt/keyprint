"""Bound execution report contract; never relabel the frozen reference.

Payload validators are shared, but execution identity and schema are distinct.
The original reference specification is reconstructed and validated before the
bounded-reference or fast envelope is admitted. Empirical acceptance does not
transfer between them.
"""
import copy
from . import fast_mlx
from .._engine.research.keyprint_v3_reporting_contract import (
    _keys, _json, _hash, _digest, _target, _scorer, _literal, _generation,
    _verification, _error, INTERPRETATION, VERSION,
)

SCHEMA = "keyprint.experimental-fast-report.v1"


def _experimental_target(raw):
    _keys(raw, ("version", "runtime_profile_sha256", "score_namespace_sha256",
                "max_steps", "deployment_calibrated", "specification"))
    _json(raw)
    if raw["version"] == "keyprint-mlx-sparse-experimental-v1":
        expected_execution = fast_mlx.execution_specification()
    elif raw["version"] == "keyprint-mlx-bounded-reference-v1":
        from ..backends.mlx_bounded import execution_specification
        expected_execution = execution_specification()
    elif raw["version"] == "keyprint-mlx-native-experimental-v2":
        from .native_mlx import execution_specification
        expected_execution = execution_specification()
    else:
        raise ValueError("known bound execution target required")
    if raw["deployment_calibrated"] is not False:
        raise ValueError("uncalibrated bound target required")
    spec = raw["specification"]
    if type(spec) is not dict or spec.get("version") != raw["version"] or _hash(raw["runtime_profile_sha256"]) != _digest(spec):
        raise ValueError("experimental specification digest mismatch")
    execution = spec.get("execution")
    if type(execution) is not dict or execution != expected_execution:
        raise ValueError("experimental execution source mismatch")
    reference = copy.deepcopy(raw)
    reference["version"] = VERSION
    reference["specification"]["version"] = VERSION
    reference["specification"].pop("execution")
    reference["runtime_profile_sha256"] = reference["specification"].pop("reference_runtime_sha256")
    _, engine = _target(reference)
    result = {key: copy.deepcopy(raw[key]) for key in raw if key != "specification"}
    result["specification_digest_verified"] = True
    result["caller_identity_authentication"] = "not_performed"
    return result, engine


def build_report(kind, *, target_identity, payload, scorer_identity=None):
    """Format caller-provided evidence once; no detection/re-evaluation occurs."""
    if kind not in ('literal_diagnostic', 'generation_trace', 'verification', 'error'):
        raise ValueError('unknown report kind')
    _json(payload)
    target, engine = _experimental_target(target_identity)
    scorer = _scorer(scorer_identity, target, engine) if scorer_identity is not None else None
    if kind == 'literal_diagnostic':
        if scorer is None: raise ValueError('literal diagnostic requires a distinct scorer identity')
        data = _literal(payload, scorer)
    elif kind == 'generation_trace':
        if target_identity['version'] == 'keyprint-mlx-native-experimental-v2':
            from .native_reporting import generation
            data = generation(payload, target, scorer)
        else:
            data = _generation(payload, target, scorer)
    else:
        if scorer is not None: raise ValueError('non-scoring reports cannot claim a scorer identity')
        data = _verification(payload, target) if kind == 'verification' else _error(payload, target)
    schema = "keyprint.bounded-reference-report.v1" if target_identity["version"] == "keyprint-mlx-bounded-reference-v1" else SCHEMA
    if target_identity["version"] == "keyprint-mlx-native-experimental-v2":
        schema = "keyprint.experimental-native-report.v2"
    result = {'schema': schema, 'kind': kind, 'integration_status': 'standalone_unintegrated_component',
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
