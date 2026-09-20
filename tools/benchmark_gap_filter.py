"""Compare complete filters against a frozen oracle; helper timing only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np


def load(path, suffix):
    spec = importlib.util.spec_from_file_location('keyprint.experimental._' + suffix, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.partition_support_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    from keyprint_native import NativePRF
    from keyprint._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter
    native = NativePRF()
    baseline = load(args.baseline, 'baseline')
    candidate = load(args.candidate, 'candidate')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    groups = ('normal', 'rounded', 'equal', 'sparse', 'tight_gap', 'huge_temperature')
    plan = {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
            'baseline_sha256': sha(args.baseline), 'candidate_sha256': sha(args.candidate),
            'native_identity': native.identity, 'width': 151936, 'mapped_width': 151669,
            'top_k': 100, 'repeats': 30, 'groups': groups, 'seed': 20260920,
            'order': 'alternate baseline/candidate first by repeat',
            'timed_scope': 'complete filter, including validation, diagnostics and immutable output',
            'failure_rule': 'all bytes, IDs, diagnostics and policy identity must equal frozen reference; no replacement or trimming'}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    rng = np.random.default_rng(plan['seed'])
    rows = []
    for group in groups:
        for repeat in range(plan['repeats']):
            values = rng.normal(size=(1, plan['width'])).astype(np.float32)
            if group == 'rounded': values[:] = np.rint(values)
            if group == 'equal': values[:] = -0.
            if group == 'sparse': values[0, 100:] = -np.inf
            settings = dict(top_k=100, temperature=1e308 if group == 'huge_temperature' else .7,
                            max_logit_gap=.1 if group == 'tight_gap' else 600.)
            expected = stable_support_filter(values, **settings)
            original = values.tobytes()
            row = {'group': group, 'repeat': repeat, 'settings': settings}
            operations = [('baseline', baseline), ('candidate', candidate)]
            if repeat % 2: operations.reverse()
            for name, function in operations:
                start = time.perf_counter()
                result = function(values, **settings, native=native)
                row[name + '_seconds'] = time.perf_counter() - start
                assert result.filtered_logits.tobytes() == expected.filtered_logits.tobytes(), (group, repeat, name, 'bytes')
                assert result.admitted_token_ids == expected.admitted_token_ids
                assert result.diagnostics == expected.diagnostics and result.identity == expected.identity
                assert not result.filtered_logits.flags.writeable and values.tobytes() == original
            rows.append(row)
            (args.output / 'rows.json').write_text(json.dumps(rows, indent=2))
    summary = {'status': 'pass', 'complete_filter_comparisons': len(rows),
               'median_candidate_over_baseline': {group: statistics.median(
                   r['candidate_seconds'] / r['baseline_seconds'] for r in rows if r['group'] == group)
                   for group in groups}, 'sdk_serving_acceptance': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
