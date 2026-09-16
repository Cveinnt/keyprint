"""Create a namespaced derivative; never modify the released reference bundle.

Only module imports, the legacy package location, and resource lookup change.
The manifest records source and derivative hashes. New execution identities are
expected; numerical/scoring equivalence is tested separately.
"""
from pathlib import Path
import hashlib
import json
import shutil
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'sdk/keyprint_v3/_bundle'
TARGET = ROOT / 'src/keyprint/_engine'
LEGACY = SOURCE / 'release/keyprint-0.0.3rc1/installed-audit/keyprint'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise SystemExit('Port already exists; edit intentionally and update provenance after review.')
    shutil.copytree(LEGACY, TARGET / 'legacy', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(SOURCE / 'research', TARGET / 'research', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (TARGET / '__init__.py').write_text('"""Private namespaced reference engine; see port-manifest.json."""\n')
    (TARGET / 'research/__init__.py').write_text('"""Private research implementation."""\n')
    mapping = {}
    for p in (TARGET / 'legacy').rglob('*'):
        if p.is_file(): mapping[p] = LEGACY / p.relative_to(TARGET / 'legacy')
    for p in (TARGET / 'research').rglob('*'):
        if p.is_file() and p.name != '__init__.py' or p.is_file() and p.parent != TARGET / 'research':
            original = SOURCE / 'research' / p.relative_to(TARGET / 'research')
            if original.exists(): mapping[p] = original
    for p in (TARGET / 'research').rglob('*.py'):
        s = p.read_text()
        # Public package no longer competes with the legacy namespace. All
        # research imports resolve normally under keyprint._engine.research.
        s = re.sub(r'(?m)^(\s*)from (keyprint_[a-z0-9_]+(?:\.[a-z0-9_]+)*) import ', r'\1from keyprint._engine.research.\2 import ', s)
        s = re.sub(r'(?m)^(\s*)import (keyprint_[a-z0-9_]+) as (\w+)', r'\1from keyprint._engine.research import \2 as \3', s)
        if p.name == 'adapter.py' and p.parent.name == 'keyprint_candidate_v2':
            s = s.replace('import sys\n', '')
            s = s.replace('INSTALLED = ROOT / "release/keyprint-0.0.3rc1/installed-audit"', 'INSTALLED = ROOT')
            s = s.replace('sys.path.insert(0, str(INSTALLED))\n', '')
            s = s.replace('import keyprint as inherited_sdk', 'from keyprint._engine import legacy as inherited_sdk')
            s = s.replace('INSTALLED / "keyprint/__init__.py"', 'INSTALLED / "legacy/__init__.py"')
            s = s.replace('INSTALLED / "keyprint"', 'INSTALLED / "legacy"')
            s = s.replace('from keyprint._impl.', 'from keyprint._engine.legacy._impl.')
        p.write_text(s)
    p = TARGET / 'legacy/verify.py'
    p.write_text(p.read_text().replace('files("keyprint")', 'files(__package__)'))
    files = {str(p.relative_to(TARGET)): {'source': str(o.relative_to(ROOT)), 'source_sha256': digest(o), 'ported_sha256': digest(p)} for p,o in sorted(mapping.items())}
    manifest = {'version': 1, 'kind': 'namespaced_derivative_not_frozen_release',
                'reference_release': 'keyprint-research-v3==0.0.4rc3',
                'changes': ['qualify research imports', 'private legacy package path', 'remove sys.path mutation', 'private importlib.resources lookup'],
                'empirical_acceptance_transfers': False, 'files': files}
    (TARGET / 'port-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__ == '__main__': main()
