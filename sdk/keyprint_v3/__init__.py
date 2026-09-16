"""Experimental v3 API; separately bound public reporting around a frozen core."""
from pathlib import Path
import hashlib
import importlib
import json
import sys

__version__ = '0.0.4rc4'
_BASE = Path(__file__).resolve().parent
_BUNDLE = _BASE / '_bundle'


def _load():
    manifest = json.loads((_BASE / 'bundle-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (_BUNDLE / relative).resolve()
        if not path.is_relative_to(_BUNDLE) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Keyprint bundled source integrity mismatch')
    # The frozen adapter explicitly binds this inherited namespace. Refuse an
    # already imported incompatible installation instead of replacing modules.
    names = ('keyprint', 'keyprint_candidate_v2', 'keyprint_candidate_v3',
             'keyprint_candidate_v3_caller', 'keyprint_exact_categorical_v2',
             'keyprint_reporting_v2', 'keyprint_stable_support_filter_v3',
             'keyprint_v3_reporting_contract', 'keyprint_v3_public_api', 'keyprint_v3_public_api_rc2')
    for name in names:
        existing = sys.modules.get(name)
        if existing is not None:
            source = getattr(existing, '__file__', None)
            if source is None or not Path(source).resolve().is_relative_to(_BUNDLE):
                raise RuntimeError('Conflicting Keyprint research import; use a fresh Python process')
    sys.path.insert(0, str(_BUNDLE / 'research'))
    return importlib.import_module('keyprint_v3_public_api_rc2')


_api = _load()
PublicCandidate = _api.PublicCandidate
PublicPipeline = _api.PublicPipeline
PublicReportError = _api.PublicReportError
# Export the existing class so the frozen caller's journal type check is preserved.
DurableJournal = importlib.import_module('keyprint_candidate_v3_caller').DurableJournal
__all__ = ['PublicCandidate', 'PublicPipeline', 'PublicReportError', 'DurableJournal']
