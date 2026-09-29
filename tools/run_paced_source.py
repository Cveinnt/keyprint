"""Separately identified, guarded paced-candidate inference on frozen source tasks.

Plan-only performs no model load. Preflight uses the first original pair/eight
tokens; study retains all original 128 attempts with their original 768-token cap.
Source data, keys and generated text stay in private receipts outside the repo.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import uuid

from paced_inference import generate, save, audit_output
from paced_source_session import Config, PacedProfile, policy_spec
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import replay_events


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def raw_counts(profile, key, token_ids):
    events = replay_events(profile, key, token_ids)
    bits = [bit for event in events if event.eligible for bit in event.bits]
    return {'events': sum(e.eligible for e in events), 'ones': sum(bits), 'trials': len(bits),
            'detector_calibrated': False, 'token_path_only': True}


def validate_preflight(path, plan):
    before = json.loads((path / 'public/plan.json').read_text())
    finished = json.loads((path / 'public/results.json').read_text())
    same = ('scripts_sha256', 'model_assets', 'policy', 'sdk_source_sha256', 'key_sha256',
            'original_plan_sha256', 'original_runs_sha256', 'dependencies')
    if (before['stage'] != 'preflight' or any(before[k] != plan[k] for k in same)
            or not before['model_load_authorized_in_this_invocation']
            or before['schedule'] != plan['schedule'][:2]
            or before['settings'] != dict(plan['settings'], max_tokens=8)
            or finished['plan_sha256'] != digest(path / 'public/plan.json')
            or finished['attempts'] != 2 or finished['successful_runs'] != 2 or finished['audited_runs'] != 2
            or finished['runs_sha256'] != digest(path / 'private/runs.json')
            or finished['identity_sha256'] != digest(path / 'public/identity.json')):
        raise ValueError('Paced preflight is incomplete or from another candidate')
    prior_rows = json.loads((path / 'private/runs.json').read_text())
    if len(prior_rows) != 2: raise ValueError('Paced preflight rows missing')
    for r, expected in zip(prior_rows, before['schedule']):
        folder = path / 'private' / r['review_id']
        if (any(r[k] != expected[k] for k in ('case', 'key_slot', 'condition'))
                or digest(folder / 'result.json') != r['result_sha256']
                or digest(folder / 'journal.jsonl') != r['journal_sha256']
                or 'error_type' in r or 'decode_error' in r or 'audit_error' in r
                or r['audit']['verified_draws'] != len(r['committed_token_ids'])
                or not 1 <= len(r['committed_token_ids']) <= 8
                or r['completion'] not in ('eos', 'cap')
                or (r['completion'] == 'cap' and len(r['committed_token_ids']) != 8)):
            raise ValueError('Paced preflight receipts changed')
    return json.loads((path / 'public/identity.json').read_text())['profile_sha256']


def inputs(root, prior, stage):
    read = lambda p: json.loads(p.read_text())
    public, private = root / 'public', root / 'private'
    original = read(public / 'plan.json'); identity = read(public / 'identity.json')
    rows = read(private / 'runs.json'); audit = read(public / 'receipt-audit.json')
    cases = read(private / 'cases.json'); rubrics = read(private / 'rubrics.json')
    from validate_source_grounded import schedule
    expected = schedule(cases)
    signature = lambda r: (r['case'], r['key_slot'], r['condition'])
    if (len(rows) != 128 or original['schedule'] != expected
            or [signature(r) for r in rows] != [signature(r) for r in expected]
            or audit['verified'] != 128 or audit['failures']
            or audit['runs_sha256'] != digest(private / 'runs.json')
            or audit['plan_sha256'] != digest(public / 'plan.json')):
        raise ValueError('Complete original schedule and receipt audit required')
    for name in ('cases.json', 'rubrics.json'):
        if digest(private / name) != original['rubrics_commitment'][name]:
            raise ValueError('Frozen source inputs or rubric changed')
    keys = [(prior / f'private/key-{i}').read_bytes() for i in range(4)]
    if any(len(k) != 32 for k in keys) or [hashlib.sha256(k).hexdigest() for k in keys] != original['key_sha256']:
        raise ValueError('Original keys changed')
    prompts = {}
    for row in rows:
        p = private / row['review_id'] / 'journal.jsonl'
        if digest(p) != row['journal_sha256']: raise ValueError('Original journal changed')
        with p.open() as f: frame = json.loads(f.readline())
        if frame['sequence'] != 0 or frame['previous_sha256'] != '0'*64:
            raise ValueError('Original journal start changed')
        ids = frame['event']['prompt_token_ids']
        if row['case'] in prompts and prompts[row['case']] != ids:
            raise ValueError('Original prompts differ within case')
        prompts[row['case']] = ids
    if stage not in ('preflight', 'study'): raise ValueError('Unknown stage')
    selected = expected[:2] if stage == 'preflight' else expected
    return original, identity, selected, prompts, keys, cases, rubrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prior', 'model', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stage', choices=('preflight', 'study'), required=True)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--preflight', type=Path)
    args = p.parse_args()
    if not args.plan_only and (str(os.getppid()) != os.environ.get('KEYPRINT_WATCHDOG_PID')
                              or 'KEYPRINT_MLX_MEMORY_LOG' not in os.environ):
        raise RuntimeError('External watchdog and guarded MLX wrapper required')
    original, identity, selected, prompts, keys, cases, rubrics = inputs(args.root, args.prior, args.stage)
    import keyprint
    from keyprint.experimental.wide_mlx import verify_assets, NFCWideByteLevelBinding
    package = Path(keyprint.__file__).parent
    sdk = lambda: {str(f.relative_to(package)): digest(f) for f in package.rglob('*.py')}
    if sdk() != original['sdk_source_sha256'] or verify_assets(args.model) != identity['model_assets']:
        raise ValueError('Frozen SDK/model differs')
    for name, expected in identity['dependencies'].items():
        if importlib.metadata.version(name) != expected: raise ValueError('Frozen runtime differs')
    helpers = ('run_paced_source.py', 'paced_inference.py', 'paced_source_session.py',
               'paced_integer_kernel.py', 'replay_wide_mlx.py', 'validate_source_grounded.py',
               'memory_watchdog.py', 'mlx_guarded_worker.py')
    hashes = {n: digest(Path(__file__).with_name(n)) for n in helpers}
    max_tokens = 8 if args.stage == 'preflight' else 768
    plan = {'schema': 'keyprint.paced-native-study.v1', 'stage': args.stage,
            'original_plan_sha256': digest(args.root / 'public/plan.json'),
            'original_runs_sha256': digest(args.root / 'private/runs.json'),
            'scripts_sha256': hashes, 'sdk_source_sha256': sdk(), 'model_assets': identity['model_assets'],
            'dependencies': identity['dependencies'], 'key_sha256': original['key_sha256'],
            'schedule': selected, 'policy': policy_spec(), 'settings': {
                'temperature': .7, 'top_k': 100, 'max_tokens': max_tokens, 'profile_max_steps': 1024},
            'randomness': 'Fresh OS SystemRandom bits; every attempted valid draw retained; pairs share prompts, not randomness',
            'purpose': 'general; summarization is not proofreading and gets no implicit source-copy routing',
            'scope': 'Exploratory candidate, all old failures retained; no causal quality, calibrated power or SDK promotion',
            'failure_policy': 'One attempt per scheduled row, no retries/repair; caps/errors retained, infrastructure abort leaves incomplete cohort',
            'data_policy': 'Source passages and derived output remain private, separate from MIT SDK',
            'model_load_authorized_in_this_invocation': not args.plan_only}
    prior_profile = None
    if args.stage == 'study' and not args.plan_only:
        if args.preflight is None: raise ValueError('Completed paced preflight required')
        prior_profile = validate_preflight(args.preflight, plan)
        plan['preflight_results_sha256'] = digest(args.preflight / 'public/results.json')
    args.output.mkdir(mode=0o700)
    public, private = args.output / 'public', args.output / 'private'
    public.mkdir(); private.mkdir(mode=0o700)
    save(public / 'plan.json', plan)
    save(private / 'cases.json', cases); save(private / 'rubrics.json', rubrics)
    save(private / 'prompt-ids.json', prompts)
    if args.plan_only:
        print(json.dumps({'planned_attempts': len(selected), 'model_loaded': False, 'stage': args.stage})); return
    import mlx.core as mx
    if mx.set_cache_limit(0) != 0: raise RuntimeError('MLX cache must already be disabled')
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    import numpy as np
    model, tokenizer = load(str(args.model), tokenizer_config={'trust_remote_code': False, 'local_files_only': True})
    if set(tokenizer.eos_token_ids) != {248046}: raise ValueError('Runtime EOS differs')
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(), vocabulary_size=248320,
        special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    profile = PacedProfile(binding.pieces, tokenizer_identity=binding.digest,
        eos_ids=binding.eos_ids, config=Config(max_steps=1024))
    if prior_profile is not None and prior_profile != profile.digest:
        raise ValueError('Paced preflight profile differs from loaded binding')
    save(public / 'identity.json', {'profile_sha256': profile.digest, 'binding_sha256': binding.digest})
    outcomes = []; rng = random.SystemRandom()
    for index, attempt in enumerate(selected):
        row = dict(attempt, review_id=uuid.uuid4().hex[:12]); cache = make_prompt_cache(model)
        def forward(ids):
            logits = model(mx.array([ids]), cache=cache)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            return np.array(logits)
        try:
            result = generate(profile=profile, binding=binding, key=keys[row['key_slot']],
                condition=row['condition'], prompt_ids=prompts[row['case']], forward=forward,
                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                random_bits=rng.getrandbits, output=private / row['review_id'], max_tokens=max_tokens)
            row.update(result)
            row['result_sha256'] = digest(private / row['review_id'] / 'result.json')
            if 'error_type' not in row and 'decode_error' not in row:
                try: row['audit'] = audit_output(private / row['review_id'], profile, keys[row['key_slot']], binding=binding)
                except Exception as error:
                    row['audit_error'] = {'type': type(error).__name__, 'message': str(error)}
            row['raw_counts'] = [raw_counts(profile, keys[slot], row['committed_token_ids'])
                                 for slot in (row['key_slot'], (row['key_slot'] + 1) % 4)]
        finally:
            cache.clear(); mx.synchronize(); mx.clear_cache()
        outcomes.append(row); save(private / 'runs.json', outcomes)
        print(f"Paced {len(outcomes)}/{len(selected)}; ending={row['completion']}; error={row.get('error_type')}", flush=True)
    if sdk() != plan['sdk_source_sha256'] or any(digest(Path(__file__).with_name(n)) != h for n,h in hashes.items()):
        raise ValueError('SDK or candidate changed during inference')
    review = [{'review_id': r['review_id'], 'case': next(c for c in cases if c['id'] == r['case']),
               'rubric': next(c for c in rubrics if c['id'] == r['case']), 'text': r.get('text'),
               'completion': r['completion'], 'error_type': r.get('error_type'),
               'decode_error': r.get('decode_error')} for r in outcomes]
    rng.shuffle(review); save(private / 'blind-review.json', review)
    save(public / 'results.json', {'schema': 'keyprint.paced-native-study-results.v1',
        'plan_sha256': digest(public / 'plan.json'), 'runs_sha256': digest(private / 'runs.json'),
        'identity_sha256': digest(public / 'identity.json'),
        'attempts': len(outcomes), 'successful_runs': sum('error_type' not in r and 'decode_error' not in r for r in outcomes),
        'audited_runs': sum('audit' in r for r in outcomes),
        'eos': sum(r['completion'] == 'eos' for r in outcomes),
        'caps': sum(r['completion'] == 'cap' for r in outcomes),
        'quality_acceptance': False, 'detector_calibrated': False, 'launch_ready': False})


if __name__ == '__main__': main()
