"""Developer-only selector comparison; not serving or SDK acceptance."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from partition_topk import select as select_top_k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    lib = ctypes.CDLL(str(args.library.resolve()))
    call = lib.keyprint_select_f32
    call.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                     ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    def native(scores, k):
        if np.all(scores == scores[0]):
            return np.arange(k)
        data = scores.astype(np.float32).tobytes()
        output = (ctypes.c_uint32 * k)()
        if call(data, len(data), k, output, ctypes.sizeof(output)) != 1:
            raise RuntimeError('native selection failed')
        return np.array(output, dtype=np.int64)
    plan = {'scope': __doc__, 'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases': '30 each: full-width normal, rounded, equal; top-k 100; alternating order',
            'comparison': 'Existing exact partition selector, inputs are exact float64 conversions from float32',
            'native_timed_scope': 'equal-score shortcut; otherwise float32 conversion, bytes copy, FFI, bounded heap, selected sort, result conversion',
            'acceptance': 'Every order must equal a full lexsort oracle; helper timing only'}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    rows = []
    rng = np.random.default_rng(20260920)
    ids = np.arange(151669)
    for kind in ('normal', 'rounded', 'equal'):
        for repeat in range(30):
            scores = rng.normal(size=len(ids)).astype(np.float32).astype(np.float64)
            if kind == 'rounded': scores = np.rint(scores)
            if kind == 'equal': scores[:] = 0
            expected = np.lexsort((ids, -scores))[:100]
            row = {'kind': kind, 'repeat': repeat}
            operations = [('partition', lambda: select_top_k(ids, scores, 100)),
                          ('native', lambda: native(scores, 100))]
            if repeat % 2: operations.reverse()
            for name, operation in operations:
                start = time.perf_counter()
                actual = operation()
                row[name+'_seconds'] = time.perf_counter()-start
                assert np.array_equal(expected, actual), (kind, repeat, name)
            rows.append(row)
            (args.output/'rows.json').write_text(json.dumps(rows, indent=2))
    result = {'status': 'pass', 'orders_verified': len(rows),
              'median_native_over_partition': {kind: statistics.median(
                  r['native_seconds']/r['partition_seconds'] for r in rows if r['kind']==kind)
                  for kind in ('normal', 'rounded', 'equal')}}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
