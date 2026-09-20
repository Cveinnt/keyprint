"""Reconcile matched-revision timing against retained journals and reports.

Checks fixture random draws, byte hashes, normalized journal events, exact text,
sampling records, token counts and independent point estimates. Does not replay
model logits, establish OS isolation or grant production serving acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from benchmark_revision import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_target(identity):
    """Verify the full worker specification, then its documented compact view."""
    expected = {'version', 'runtime_profile_sha256', 'score_namespace_sha256',
                'max_steps', 'deployment_calibrated', 'specification'}
    if set(identity) != expected or identity['deployment_calibrated'] is not False:
        raise ValueError('Unexpected or calibrated worker identity')
    if not isinstance(identity['specification'], dict) or identity['specification'].get('version') != identity['version']:
        raise ValueError('Worker specification version differs')
    encoded = json.dumps(identity['specification'], sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != identity['runtime_profile_sha256']:
        raise ValueError('Worker specification digest differs')
    return {**{k: v for k, v in identity.items() if k != 'specification'},
            'specification_digest_verified': True, 'caller_identity_authentication': 'not_performed'}


def check_journal(path, seed, expected_identity=None):
    previous, events = '0' * 64, []
    rng = random.Random(seed)
    for i, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        row = json.loads(line)
        if row['sequence'] != i or row['previous_sha256'] != previous:
            raise ValueError('Journal chain differs')
        previous = hashlib.sha256(line).hexdigest()
        event = row['event']
        for name, value in (expected_identity or {}).items():
            if name in event and event[name] != value:
                raise ValueError('Journal execution identity differs')
        if event['kind'] == 'random_bits_returned':
            if int(event['value_decimal']) != rng.getrandbits(event['bits']):
                raise ValueError('Journal draw differs from the declared fixture stream')
        events.append({k: v for k, v in event.items()
                       if k not in ('runtime_profile_sha256', 'caller_sha256')})
    return events


def audit(root):
    public = root / 'public'
    plan = json.loads((public / 'plan.json').read_text())
    summary = json.loads((public / 'summary.json').read_text())
    tools = Path(__file__).parent
    if (plan['script_sha256'] != sha(tools / 'benchmark_revision.py')
            or plan['analysis_sha256'] != sha(tools / 'benchmark_serving.py')
            or plan['cases_sha256'] != sha(tools / 'inference_cases.json')):
        raise ValueError('Study sources differ from the declaration')
    if summary['status'] != 'completed' or summary['failure'] is not None:
        raise ValueError('Incomplete studies cannot receive a completed audit')
    if summary['requests'] != plan['measured_requests'] or summary['warmups'] != 4:
        raise ValueError('Declared request count differs')
    if summary['worker_exit_codes'] != {'baseline': 0, 'candidate': 0}:
        raise ValueError('Both workers must have shut down cleanly')
    identities = {arm: json.loads((public / f'{arm}-identity.json').read_text())
                  for arm in ('baseline', 'candidate')}
    for arm, identity in identities.items():
        if not identity['ready'] or identity['package_files'] != plan['packages'][arm]:
            raise ValueError('Worker package declaration differs')
    rows, journals, tokens = [], {}, 0
    for warmup in (True, False):
        for repeat in (-1,) if warmup else range(plan['repeats']):
            for ci, case in enumerate(plan['cases'][:1] if warmup else plan['cases']):
                for co, condition in enumerate(('ordinary', 'marked')):
                    for arm in ('baseline', 'candidate'):
                        name = f'{"warmup-" if warmup else ""}{case["id"]}-{repeat}-{condition}-{arm}'
                        row = json.loads((public / f'{name}.json').read_text())
                        seed = 20260920 + repeat * 100 + ci * 2 + co
                        cap = 32 if warmup else case['max_tokens']
                        if (row['id'] != name or row['seed'] != seed or row['max_tokens'] != cap
                                or row['arm'] != arm or row['condition'] != condition
                                or row['repeat'] != repeat or row['case'] != case['id'] or 'error' in row):
                            raise ValueError('Retained request differs from its declaration')
                        directory = root / name
                        if (sha(directory / 'report.json') != row['report_sha256']
                                or sha(directory / 'journal.jsonl') != row['journal_sha256']):
                            raise ValueError('Receipt bytes changed')
                        saved = json.loads((directory / 'report.json').read_text())
                        report, payload = saved['report'], saved['report']['payload']
                        if (report['kind'] != 'generation_trace'
                                or report['target_identity'] != report_target(identities[arm]['identity'])
                                or report['rendered_carriers']['visible_text'] != row['text']
                                or payload['committed_token_ids'] != row['token_ids']
                                or payload['completion'] != row['completion']
                                or payload['sampling_records'] != row['sampling_records']
                                or saved['reservations'] != row['reservations']):
                            raise ValueError('Displayed output or work differs from its report')
                        events = check_journal(directory / 'journal.jsonl', seed, {
                            'runtime_profile_sha256': identities[arm]['identity']['runtime_profile_sha256'],
                            'caller_sha256': plan['packages'][arm]['keyprint/experimental/fast_caller.py']})
                        if (len(row['token_ids']) != row['completion_tokens']
                                or len(row['token_ids']) != sum(e['kind'] == 'committed_step' for e in events)
                                or events[-1]['outcome'] != row['completion']
                                or events[0]['condition'] != condition or events[0]['max_tokens'] != cap):
                            raise ValueError('Token count or completion differs from its journal')
                        identity = (warmup, case['id'], repeat, condition)
                        if arm == 'baseline': journals[identity] = events
                        elif journals.pop(identity) != events:
                            raise ValueError('Revision journal events differ beyond their bound source identities')
                        tokens += len(row['token_ids'])
                        if not warmup: rows.append(row)
    result = analyze(rows, [c['id'] for c in plan['cases']])
    if result != summary['analysis']:
        raise ValueError('Timing summary differs from the retained complete sample')
    for condition in ('ordinary', 'marked'):
        selected = [r for r in rows if r['condition'] == condition]
        times = {(r['case'], r['repeat'], r['arm']): r['seconds'] for r in selected}
        ratios = [math.log(times[c['id'], repeat, 'candidate'] / times[c['id'], repeat, 'baseline'])
                  for c in plan['cases'] for repeat in range(plan['repeats'])]
        expected = math.exp(statistics.fmean(ratios))
        for metric in ('request_latency', 'seconds_per_committed_token'):
            actual = result['candidate_over_baseline'][condition][metric]['geometric_mean_ratio']
            if not math.isclose(actual, expected, rel_tol=1e-14, abs_tol=0.):
                raise ValueError('Independent matched-path point estimate differs')
    return {'status': 'pass', 'requests': len(rows), 'warmups': 4, 'tokens': tokens,
            'scope': __doc__, 'audit_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    (args.run / 'public' / 'integrity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
