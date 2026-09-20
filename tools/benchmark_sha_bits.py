"""Development-only screen for producing all layer bits in one Python call.

The installed SDK is unchanged. Compare exact bits against its existing SHA
templates; helper timing is not serving overhead or production acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from keyprint.experimental.hmac_context import SHAContext


def layer_bits(context, suffix):
    bits = []
    append = bits.append
    outer_copy = context.outer.copy
    for template in context.templates:
        inner = template.copy()
        inner.update(suffix)
        outer = outer_copy()
        outer.update(inner.digest())
        append(outer.digest()[0] & 1)
    return tuple(bits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    plan = {'sizes': [1, 10, 64, 100, 1000], 'repeats': 20, 'layers': 30,
            'seed': 20260920, 'key_count': 3,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'order': 'Alternate reference/prototype first by repeat plus key index', 'scope': __doc__}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    rng = random.Random(plan['seed'])
    keys = [rng.randbytes(32) for _ in range(plan['key_count'])]
    rows, comparisons = [], 0
    for size in plan['sizes']:
        for key_index, key in enumerate(keys):
            for repeat in range(plan['repeats']):
                context = SHAContext(key, rng.randbytes(repeat * 7), plan['layers'])
                labels = [rng.randbytes(1 + i % 64) for i in range(size)]
                suffixes = [len(label).to_bytes(8, 'big') + label for label in labels]
                def reference():
                    return [tuple(context.digest(layer, suffix)[0] & 1
                                  for layer in range(plan['layers'])) for suffix in suffixes]
                def prototype():
                    return [layer_bits(context, suffix) for suffix in suffixes]
                timings, values = {}, {}
                order = [('reference', reference), ('prototype', prototype)]
                if (repeat + key_index) % 2: order.reverse()
                for name, fn in order:
                    started = time.perf_counter_ns()
                    values[name] = fn()
                    timings[name] = time.perf_counter_ns() - started
                if values['reference'] != values['prototype']:
                    raise AssertionError('Layer bits differ')
                comparisons += size * plan['layers']
                rows.append({'size': size, 'key_index': key_index, 'repeat': repeat,
                             'nanoseconds': timings})
    result = {'status': 'pass', 'bit_comparisons': comparisons,
              'median_prototype_over_reference_by_size': {
                  str(size): statistics.median(r['nanoseconds']['prototype'] / r['nanoseconds']['reference']
                                               for r in rows if r['size'] == size)
                  for size in plan['sizes']},
              'sdk_changed': False, 'serving_acceptance': False, 'rows': rows}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
