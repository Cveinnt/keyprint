"""Compare raw dense SHA-256 with a proposed lossless vector commitment.

Synthetic vector helper screen only. This changes hash encoding and is not
integrated into SDK receipts, caller parity or production cost acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from vector_commitment import encode_vector, decode_vector, vector_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan = {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
            'codec_sha256': sha(Path(__file__).with_name('vector_commitment.py')),
            'seed': 20260920, 'repeats': 20, 'supports': [1, 10, 100, 512, 4096, 'dense'],
            'widths': {'probability': 151669, 'filtered_logits': 151936},
            'timed_scope': 'Validation/encoding plus SHA-256 versus existing array.tobytes plus SHA-256; all results retained',
            'order': 'alternate operation order each repeat; correctness checked before timing',
            'versions': {'numpy': importlib.metadata.version('numpy'), 'python': platform.python_version()},
            'failure_rule': 'Retain every row; any changed bit fails the screen; no retries or trimmed outcomes'}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    rng = np.random.default_rng(plan['seed'])
    rows = []
    for kind, width in plan['widths'].items():
        for support in plan['supports']:
            count = width if support == 'dense' else support
            for repeat in range(plan['repeats']):
                values = np.full(width, 0. if kind == 'probability' else -np.inf)
                ids = rng.choice(width, count, replace=False)
                if kind == 'probability':
                    weights = rng.uniform(.1, 1, count); weights /= weights.sum()
                else:
                    weights = -rng.uniform(0, 30, count); weights[0] = -0.
                values[ids] = weights
                if kind == 'filtered_logits': values = values.reshape(1, -1)
                original = values.tobytes()
                packet = encode_vector(values, kind)
                assert decode_vector(packet).values.tobytes() == original
                raw_hash = hashlib.sha256(original).hexdigest()
                new_hash = hashlib.sha256(packet).hexdigest()
                row = {'kind': kind, 'support': support, 'repeat': repeat,
                       'dense_bytes': len(original), 'encoded_bytes': len(packet),
                       'dense_sha256': raw_hash, 'encoded_sha256': new_hash}
                operations = [('dense', lambda: hashlib.sha256(values.tobytes()).hexdigest()),
                              ('candidate', lambda: vector_digest(values, kind))]
                if repeat % 2: operations.reverse()
                for name, call in operations:
                    start = time.perf_counter(); actual = call()
                    row[name+'_seconds'] = time.perf_counter()-start
                    assert actual == (raw_hash if name == 'dense' else new_hash)
                rows.append(row)
                (args.output/'rows.json').write_text(json.dumps(rows, indent=2))
    result = {'status': 'pass', 'bit_exact_roundtrips': len(rows), 'groups': []}
    for kind in plan['widths']:
        for support in plan['supports']:
            group = [r for r in rows if r['kind'] == kind and r['support'] == support]
            result['groups'].append({'kind': kind, 'support': support,
                'median_candidate_over_dense': statistics.median(r['candidate_seconds']/r['dense_seconds'] for r in group),
                'dense_bytes': group[0]['dense_bytes'], 'encoded_bytes': group[0]['encoded_bytes']})
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__': main()
