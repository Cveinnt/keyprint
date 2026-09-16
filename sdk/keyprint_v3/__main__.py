"""Offline fixture JSON retains interpretation; startup errors use stderr/nonzero."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from . import PublicCandidate, PublicReportError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('demo', 'score-fixture', 'verify'))
    parser.add_argument('--condition', choices=('ordinary', 'marked'), default='marked')
    args = parser.parse_args()
    candidate = PublicCandidate()
    try:
        if args.command == 'score-fixture':
            report = candidate.score_literal(sys.stdin.read(), bytes(range(32)))
        elif args.command == 'demo':
            with candidate.pipeline(bytes(range(32)), condition=args.condition) as pipeline:
                for _ in range(3):
                    head = np.full((1, 151936), -np.inf, dtype=np.float32)
                    head[0, 32:34] = (0, -1)
                    report = pipeline.step(head, lambda bits: 0)
                    if report['kind'] == 'error':
                        break
                else:
                    head = np.full((1, 151936), -np.inf, dtype=np.float32)
                    head[0, 151645] = 0
                    report = pipeline.step(head, lambda bits: 0)
                    if report['kind'] != 'error':
                        report = pipeline.finish()
            report['example_scope'] = 'Public fixture key, supplied logits and deterministic zero bits; no language model or detection efficacy claim.'
        else:
            manifest = Path(__file__).with_name('bundle-manifest.json')
            report = candidate.project_verification({
                'status': 'pass', 'artifact_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'evaluated_runtime_profile_sha256': candidate.core_identity['runtime_profile_sha256'],
                'checks': [{'name': 'bundled_file_sha256', 'status': 'pass'}]})
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
        return 1 if report['kind'] == 'error' else 0
    except PublicReportError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, allow_nan=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
