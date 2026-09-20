"""Reporting facade for the explicit experimental execution identity."""
import hashlib
from pathlib import Path
from threading import get_ident

from .._engine.research import keyprint_v3_public_api_rc2 as reference_public
from .._engine.research.keyprint_v3_public_api_rc2 import PublicCandidate, _digest
from .._engine.research import keyprint_v3_reporting_contract as contract
from .._engine.research import keyprint_reporting_v2 as literal_contract
from .fast_reporting import build_report
from .fast_mlx import FastCandidate
from .fast_caller import run_response as _run_response, ResponseFailure


class FastPublicCandidate(PublicCandidate):
    def __init__(self, reference):
        if type(reference) is not PublicCandidate:
            raise TypeError("An exact reference facade is required")
        reference._ready()
        self._core = FastCandidate(reference._core)
        self._owner, self._busy = get_ident(), False
        self._target = self._core.identity
        engine = self._target["specification"]["inherited_engine_identity"]
        self._scorer = {"runtime_profile_sha256": engine["runtime_profile_sha256"],
                        "score_namespace_sha256": engine["score_namespace_sha256"],
                        "runtime_max_steps": engine["max_steps"]}
        spec = {"version": "keyprint-experimental-fast-facade-v1",
                "core_runtime_profile_sha256": self._target["runtime_profile_sha256"],
                "sources": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in (__file__, reference_public.__file__, contract.__file__, literal_contract.__file__)},
                "reference_results_transfer": False}
        self._identity = {"facade_profile_sha256": _digest(spec), "specification": spec}

    def _decorate(self, report):
        report = super()._decorate(report)
        report["integration_status"] = "experimental_sparse_execution_lifecycle_qualification_pending"
        report["package_scope"] = "Experimental sparse execution; no new model-family, scientific or serving acceptance"
        return report

    def _error(self, phase, *, exc=None, failure=None, receipt=None):
        f = failure or {}
        r = receipt or {}
        code = ('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) or f.get('outcome') == 'interrupted'
            else 'journal_failure' if f.get('journal_broken')
            else 'invalid_input' if isinstance(exc, (ValueError, TypeError)) else 'runtime_failure')
        payload = {'code': code, 'phase': f.get('phase', phase),
            'committed_tokens': f.get('sampled_tokens', len(r.get('committed_token_ids', []))),
            'bit_requests': f.get('bit_requests', r.get('attempted_bit_requests', 0)),
            'bit_values_obtained': f.get('bit_values_obtained', r.get('completed_bit_returns', 0)),
            'bit_values_journaled': f.get('bit_values_journaled', 0)}
        return self._decorate(build_report('error', target_identity=self._target, payload=payload))

    def score_literal(self, text, key):
        self._ready()
        try:
            raw = self._literal_projection(self._core.score_literal(text, key))
            return self._decorate(build_report('literal_diagnostic', target_identity=self._target,
                                               scorer_identity=self._scorer, payload=raw))
        except BaseException as exc:
            return self._error('literal_diagnostic', exc=exc)

    def _generation(self, receipt, *, latest_text=None):
        if receipt['runtime'] != self._target or receipt['calibrated'] is not False:
            raise ValueError('generation target mismatch')
        final = receipt['final']
        completion = 'incomplete' if final is None else 'eos' if final['reason'] == 'declared_eos' else 'length'
        diagnostics, content, replay_status = [], None, []
        if final is not None:
            if final['runtime_profile_sha256'] != self._target['runtime_profile_sha256']:
                raise ValueError('final runtime mismatch')
            carriers = [final['visible'], final['reasoning'], *final['tools']]
            channel_names = ['visible', 'reasoning', *[f'tool_{i}' for i in range(len(final['tools']))]]
            for channel_name, carrier in zip(channel_names, carriers):
                if carrier is None: continue
                if carrier['profile_sha256'] != self._target['runtime_profile_sha256'] or carrier['deployment_calibrated'] is not False:
                    raise ValueError('carrier identity or calibration mismatch')
                score = carrier['text_score']
                replay_status.append({'channel': channel_name,
                    'availability': 'available' if score is not None else 'unavailable',
                    'reason': carrier.get('unavailable_reason') if score is None else None})
                if score is None:
                    # Completed output remains valid when literal retokenization is unavailable.
                    # Never substitute generation-path scores or zero-event counts.
                    score = {'events': None, 'ones': None, 'trials': None,
                             'random_key_null_p': None, 'underflow_floor': False}
                events = score['events']
                diagnostics.append({**self._scorer, **score,
                    'status': 'available' if events else 'unavailable',
                    'key_commitment': receipt['key_commitment'],
                    'text_sha256': hashlib.sha256(carrier['text'].encode()).hexdigest(),
                    'deployment_calibrated': False, 'verdict': None})
            content = {'visible_text': final['visible']['text'],
                'reasoning_text': final['reasoning']['text'] if final['reasoning'] else None,
                'tool_texts': [v['text'] for v in final['tools']],
                'protocol_complete': final['protocol_complete'],
                'interpret_as': 'generated_content_not_report_instructions_or_provenance'}
        payload = {'condition': receipt['condition'], 'completion': completion,
            'committed_token_ids': receipt['committed_token_ids'], 'sampling_records': receipt['sampling_records'],
            'trace_artifact_sha256': _digest(receipt), 'literal_diagnostics': diagnostics}
        report = build_report('generation_trace', target_identity=self._target,
                             scorer_identity=self._scorer if diagnostics else None, payload=payload)
        report['rendered_carriers'] = content
        report['literal_replay_status'] = replay_status
        if latest_text is not None: report['latest_emitted_text'] = latest_text
        report['trace_artifact_availability'] = 'hash_reference_only_original_private_trace_not_exported'
        return self._decorate(report)

    def run_response(self, model, prompt_ids, **settings):
        self._ready()
        self._busy = True
        captured = []
        result = None
        original = self._core.pipeline
        def observed_factory(*args, **kwargs):
            value = original(*args, **kwargs)
            captured.append(value)
            return value
        self._core.pipeline = observed_factory
        try:
            result = _run_response(self._core, model, prompt_ids, **settings)
            if len(captured) != 1: raise ValueError('caller must create exactly one pipeline')
            receipt = captured[0].receipt()
            if receipt['final'] != result['response']:
                raise ValueError('caller response and retained pipeline differ')
            return self._generation(receipt)
        except ResponseFailure as exc:
            return self._error('caller', failure=exc.failure)
        except BaseException as exc:
            receipt = captured[0].receipt() if len(captured) == 1 else None
            return self._error('caller', exc=exc, receipt=receipt,
                failure={**result, 'phase': 'caller_projection'} if result is not None else None)
        finally:
            del self._core.pipeline
            self._busy = False
            for value in captured: value.close()

    def project_verification(self, payload):
        self._ready()
        try:
            return self._decorate(build_report('verification', target_identity=self._target, payload=payload))
        except BaseException as exc:
            return self._error('verification_projection', exc=exc)
