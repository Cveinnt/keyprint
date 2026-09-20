"""Development-only native HMAC batch screen; no SDK or serving acceptance."""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import random
import statistics
import time

from native_prf.batch import BatchSHA256
from keyprint.experimental.hmac_context import SHAContext


def reference_digests(key, prefix, suffixes, layers):
    context = SHAContext(key, prefix, layers)
    return b''.join(context.digest(layer, suffix)
                    for suffix in suffixes for layer in range(layers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    native = BatchSHA256(args.library)
    plan = {'sizes': [1, 10, 64, 100, 1000], 'repeats': 20, 'layers': 30,
            'key_count': 3, 'seed': 20260922,
            'prefix_boundary_sizes': [0, 1, 55, 56, 63, 64, 65, 127, 128, 1024, 65536],
            'suffix_boundary_sizes': [0, 1, 55, 56, 63, 64, 65, 127, 128, 1024, 4096],
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'c_source_sha256': hashlib.sha256(Path(__file__).with_name('native_prf').joinpath('batch_sha256.c').read_bytes()).hexdigest(),
            'wrapper_sha256': hashlib.sha256(Path(__file__).with_name('native_prf').joinpath('batch.py').read_bytes()).hexdigest(),
            'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'openssl': native.openssl_version,
            'reference': 'installed SHAContext with context construction included',
            'order': 'alternate reference/native first by repeat plus key index',
            'scope': 'Full digest parity and helper timing, including Python packing/output copying. No model inference, SDK integration or serving claim.'}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    rng = random.Random(plan['seed'])
    keys = [rng.randbytes(32) for _ in range(plan['key_count'])]
    boundaries = 0
    for key in keys:
        for prefix_size in plan['prefix_boundary_sizes']:
            prefix = rng.randbytes(prefix_size)
            suffixes = [rng.randbytes(size) for size in plan['suffix_boundary_sizes']]
            expected = b''.join(hmac.digest(key, prefix + layer.to_bytes(4, 'big') + suffix, 'sha256')
                                for suffix in suffixes for layer in range(plan['layers']))
            assert native.digests(key, prefix, suffixes, plan['layers']) == expected
            assert reference_digests(key, prefix, suffixes, plan['layers']) == expected
            boundaries += len(suffixes) * plan['layers']
    rows, comparisons = [], 0
    for size in plan['sizes']:
        for key_index, key in enumerate(keys):
            for repeat in range(plan['repeats']):
                prefix = rng.randbytes(repeat * 7)
                labels = [rng.randbytes(1 + i % 64) for i in range(size)]
                suffixes = [len(label).to_bytes(8, 'big') + label for label in labels]
                expected = b''.join(hmac.digest(key, prefix + layer.to_bytes(4, 'big') + suffix, 'sha256')
                                    for suffix in suffixes for layer in range(plan['layers']))
                timings = {}
                functions = [('reference', reference_digests), ('native', native.digests)]
                if (repeat + key_index) % 2: functions.reverse()
                for name, fn in functions:
                    started = time.perf_counter_ns()
                    actual = fn(key, prefix, suffixes, plan['layers'])
                    timings[name] = time.perf_counter_ns() - started
                    if actual != expected:
                        raise AssertionError(f'Full digest mismatch: {name}, size={size}, repeat={repeat}')
                comparisons += size * plan['layers']
                rows.append({'size': size, 'key_index': key_index, 'repeat': repeat, 'nanoseconds': timings})
    result = {'status': 'pass', 'boundary_digest_comparisons': boundaries,
              'timing_digest_comparisons': comparisons,
              'rows': rows, 'median_native_over_reference_by_size': {
                  str(size): statistics.median(row['nanoseconds']['native'] / row['nanoseconds']['reference']
                                               for row in rows if row['size'] == size)
                  for size in plan['sizes']}, 'sdk_changed': False, 'serving_accepted': False}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
