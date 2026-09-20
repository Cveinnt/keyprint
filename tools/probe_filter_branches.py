"""Observe actual filter branches; diagnostic instrumentation, not serving cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.getprofile() is not None:
        raise RuntimeError('An existing profiler must not be replaced')
    args.output.mkdir(mode=0o700)
    public = args.output / 'public'; public.mkdir()
    from keyprint import Keyprint
    from keyprint.experimental import partition_filter
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / 'owner.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
        f.write(key)
    model = Keyprint.from_mlx(args.model, key=key, execution='experimental-native')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cases_path = Path(__file__).with_name('inference_cases.json')
    cases = json.loads(cases_path.read_text())
    plan = {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'cases': cases,
            'cases_sha256': sha(cases_path), 'identity': model.identity,
            'filter_source_sha256': sha(Path(partition_filter.__file__)),
            'instrumentation': 'Read exact filter frame locals on Python profile return; no model or filter patch',
            'branch_rule': 'Successful range path has no losses local; fallback computes losses',
            'failure_rule': 'Retain all twelve outputs and failures; no retries or replacements'}
    (public / 'plan.json').write_text(json.dumps(plan, indent=2))
    target = partition_filter.partition_support_filter.__code__
    observed = []
    def profile(frame, event, result):
        if frame.f_code is target and event == 'return':
            values = frame.f_locals
            observed.append({'status': 'returned' if result is not None else 'failed',
                'branch': 'per_token' if 'losses' in values else 'all_finite',
                'span_upper': values.get('span_upper'), 'scaled_upper': values.get('scaled_upper'),
                'eligible_count': values.get('eligible_count')})
    rows = []
    for index, case in enumerate(cases):
        for condition in (('ordinary', 'marked') if index % 2 == 0 else ('marked', 'ordinary')):
            observed = []
            row = {'case': case['id'], 'condition': condition}
            name = case['id'] + '-' + condition
            try:
                sys.setprofile(profile)
                result = model.generate(case['prompt'], condition=condition,
                    max_tokens=case['max_tokens'], output=args.output / name)
                tokens = result.report['usage']['completion_tokens']
                assert tokens == len(observed) and all(o['status'] == 'returned' for o in observed)
                row.update(status='pass', tokens=tokens,
                    fast_calls=sum(o['branch'] == 'all_finite' for o in observed),
                    fallback_calls=sum(o['branch'] == 'per_token' for o in observed),
                    report_sha256=sha(args.output / name / 'report.json'))
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            finally:
                sys.setprofile(None)
            (args.output / (name + '-branches.json')).write_text(json.dumps(observed, indent=2))
            row['observations_sha256'] = sha(args.output / (name + '-branches.json'))
            rows.append(row)
            (public / 'rows.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    summary = {'status': 'pass' if len(rows) == 12 and all(r['status'] == 'pass' for r in rows) else 'failed',
               'outputs': len(rows), **{k: sum(r.get(k, 0) for r in rows) for k in ('tokens', 'fast_calls', 'fallback_calls')},
               'scope': __doc__}
    (public / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return int(summary['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
