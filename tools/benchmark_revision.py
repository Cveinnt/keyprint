"""Matched-path comparison of two installed native MLX SDK revisions.

Public fixture key and explicit fixture draws only. Two resident, independent
workers execute sequentially with fresh request caches. All cases, failures and
caps remain recorded. This isolates caller revisions on matched token paths;
it does not measure production randomness, HTTP, batching or engine overhead.
"""
import argparse
import json
import os
from pathlib import Path
import random
import select
import subprocess
import sys
import time

from benchmark_serving import REPEATS, require_storage, sha, summarize

ARMS = ('baseline', 'candidate')
CONDITIONS = ('ordinary', 'marked')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def package_files(root):
    paths = sorted((root / 'keyprint').rglob('*'))
    return {str(p.relative_to(root)): sha(p) for p in paths
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def paired_rows(rows, case_ids):
    expected = {(case, repeat, condition, arm) for case in case_ids
                for repeat in range(REPEATS) for condition in CONDITIONS for arm in ARMS}
    by_id = {(r['case'], r['repeat'], r['condition'], r['arm']): r for r in rows}
    if len(rows) != len(expected) or set(by_id) != expected or any('error' in r for r in rows):
        raise ValueError('Every declared revision/condition pair must finish exactly once')
    pairs = []
    for case in case_ids:
        for repeat in range(REPEATS):
            for condition in CONDITIONS:
                left, right = (by_id[case, repeat, condition, arm] for arm in ARMS)
                fields = ('seed', 'prompt_ids', 'token_ids', 'text', 'completion',
                          'completion_tokens', 'sampling_records', 'reservations')
                if any(left[field] != right[field] for field in fields):
                    raise ValueError('Matched-path revisions differ in draws, tokens, text or work')
                for row in (left, right):
                    if row['completion_tokens'] != len(row['token_ids']) or row['completion_tokens'] < 1:
                        raise ValueError('Token denominator differs from the retained path')
                pairs.append((left, right))
    return pairs


def analyze(rows, case_ids):
    paired_rows(rows, case_ids)
    result = {}
    for condition in CONDITIONS:
        mapped = [{**r, 'condition': 'ordinary' if r['arm'] == 'baseline' else 'marked'}
                  for r in rows if r['condition'] == condition]
        stats = summarize(mapped, case_ids)
        # Do not expose the within-SDK 5% gate as a revision qualification.
        result[condition] = {k: stats[k] for k in ('pairs', 'seconds_per_committed_token',
                                                  'request_latency', 'uncertainty')}
    return {'candidate_over_baseline': result, 'matched_paths': True,
            'production_overhead_accepted': False, 'scope': __doc__}


def worker(args):
    import keyprint
    import mlx.core as mx
    from keyprint import Keyprint
    from keyprint._engine.research.keyprint_candidate_v3_caller import DurableJournal
    if Path(keyprint.__file__).resolve().parent.parent != args.sdk_root.resolve():
        raise ValueError('Worker did not import the requested installed SDK')
    candidate = Keyprint.from_mlx(args.model, key=bytes(range(32)), execution='experimental-native')
    backend = candidate._backend
    mx.synchronize()
    print(json.dumps({'ready': True, 'identity': candidate.identity,
                      'package_files': package_files(args.sdk_root), 'pid': os.getpid()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            return
        output = Path(request['output'])
        row = {k: request[k] for k in ('id', 'case', 'repeat', 'condition', 'arm', 'seed', 'max_tokens')}
        mx.synchronize()
        started = time.perf_counter()
        try:
            output.mkdir(mode=0o700)
            prompt = backend.encode_prompt(request['prompt'])
            reservations = {}
            def reserve(action, metadata):
                reservations[action] = reservations.get(action, 0) + 1
            with DurableJournal(output / 'journal.jsonl') as journal:
                report = candidate._candidate.run_response(
                    backend.model, prompt, key=bytes(range(32)), condition=request['condition'],
                    random_bits=random.Random(request['seed']).getrandbits,
                    journal=journal, reserve=reserve, max_tokens=request['max_tokens'],
                    max_model_calls=request['max_tokens'] + 8,
                    allow_thinking=False, allow_tools=False)
            write(output / 'report.json', {'report': report, 'reservations': reservations})
            mx.synchronize()
            row['seconds'] = time.perf_counter() - started
            if report['kind'] != 'generation_trace':
                raise ValueError('Caller returned an error report')
            payload = report['payload']
            row.update(prompt_ids=prompt, token_ids=payload['committed_token_ids'],
                       text=report['rendered_carriers']['visible_text'], completion=payload['completion'],
                       completion_tokens=len(payload['committed_token_ids']),
                       sampling_records=payload['sampling_records'], reservations=reservations,
                       report_sha256=sha(output / 'report.json'), journal_sha256=sha(output / 'journal.jsonl'))
        except Exception as exc:
            row.update(error={'type': type(exc).__name__, 'message': str(exc)},
                       seconds=time.perf_counter() - started)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def receive(process):
    if not select.select([process.stdout], [], [], 300)[0]:
        raise TimeoutError('Revision worker did not respond within 300 seconds')
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'Revision worker stopped with exit {process.poll()}')
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--idle-host-confirmed', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--sdk-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not all((args.output, args.baseline, args.candidate, args.idle_host_confirmed)):
        parser.error('baseline, candidate, output and idle-host-confirmed are required')
    free = require_storage(args.output)
    args.output.mkdir(mode=0o700)
    public = args.output / 'public'
    public.mkdir()
    cases_path = Path(__file__).with_name('inference_cases.json')
    cases = json.loads(cases_path.read_text())
    roots = {arm: getattr(args, arm).resolve() for arm in ARMS}
    plan = {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'cases': cases,
            'cases_sha256': sha(cases_path), 'analysis_sha256': sha(Path(__file__).with_name('benchmark_serving.py')),
            'packages': {arm: package_files(root) for arm, root in roots.items()},
            'key': 'PUBLIC FIXTURE ONLY: bytes(range(32))', 'repeats': REPEATS,
            'measured_requests': len(cases) * REPEATS * 4,
            'warmups': 'One retained 32-token request for each revision and condition, excluded prospectively',
            'order': 'Repeat, case, condition; baseline first when sum of indices even, candidate otherwise',
            'draws': 'Fresh Random(20260920 + repeat * 100 + case_index * 2 + condition_index) per request',
            'timed_scope': 'Prompt encoding, fresh cache, real model calls, filter/sampler, durable journal, report projection and report-file serialization',
            'excluded': 'Imports, model loading, IPC, model inspection, HTTP, public generate preflight and production crypto draw source',
            'failure_rule': 'No retries, replacements or exclusions; any failure prevents completed analysis',
            'os_isolation': False, 'resident_models': 2, 'free_bytes_before': free}
    write(public / 'plan.json', plan)
    workers, logs, rows, warmups, failure, analysis = {}, [], [], [], None, None
    try:
        for arm, root in roots.items():
            log = (args.output / f'{arm}-worker.log').open('x')
            logs.append(log)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                '--sdk-root', str(root), '--model', str(args.model.resolve())],
                env={**os.environ, 'PYTHONPATH': str(root), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
            workers[arm] = process
            ready = receive(process)
            if not ready.get('ready') or ready['package_files'] != plan['packages'][arm]:
                raise ValueError('Worker installation differs from the frozen plan')
            write(public / f'{arm}-identity.json', ready)
        def attempt(case, repeat, ci, condition, arm, warmup=False):
            name = f'{"warmup-" if warmup else ""}{case["id"]}-{repeat}-{condition}-{arm}'
            request = {'id': name, 'case': case['id'], 'repeat': repeat, 'condition': condition,
                       'arm': arm, 'seed': 20260920 + repeat * 100 + ci * 2 + CONDITIONS.index(condition),
                       'max_tokens': 32 if warmup else case['max_tokens'], 'prompt': case['prompt'],
                       'output': str((args.output / name).resolve())}
            p = workers[arm]
            p.stdin.write(json.dumps(request) + '\n')
            p.stdin.flush()
            row = receive(p)
            if any(row[k] != request[k] for k in ('id', 'case', 'repeat', 'condition', 'arm', 'seed', 'max_tokens')):
                raise ValueError('Worker response differs from its request')
            write(public / f'{name}.json', row)
            print(json.dumps({k: row[k] for k in ('id', 'seconds', 'completion_tokens', 'error') if k in row}), flush=True)
            return row
        for condition in CONDITIONS:
            for arm in ARMS:
                warmups.append(attempt(cases[0], -1, 0, condition, arm, True))
        if any('error' in row for row in warmups):
            raise ValueError('Warmup failure prevents measurement')
        for repeat in range(REPEATS):
            for ci, case in enumerate(cases):
                for co, condition in enumerate(CONDITIONS):
                    for arm in ARMS if (repeat + ci + co) % 2 == 0 else reversed(ARMS):
                        rows.append(attempt(case, repeat, ci, condition, arm))
        analysis = analyze(rows, [c['id'] for c in cases])
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        for process in workers.values():
            if process.poll() is None:
                try:
                    process.stdin.write('{"stop":true}\n'); process.stdin.flush()
                    process.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
            process.stdin.close(); process.stdout.close()
        for log in logs: log.close()
    if failure is None and any(p.returncode != 0 for p in workers.values()):
        failure = {'type': 'WorkerExitError', 'message': 'A revision worker did not shut down cleanly'}
        analysis = None
    summary = {'status': 'completed' if failure is None else 'incomplete', 'failure': failure,
               'requests': len(rows), 'warmups': len(warmups), 'analysis': analysis,
               'worker_exit_codes': {arm: p.returncode for arm, p in workers.items()}}
    write(public / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
