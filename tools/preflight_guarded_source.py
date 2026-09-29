"""Bounded resource preflight: eight recorded steps from the first two study rows.

No generation, new draws, replacement outputs or modification of the original
replay. Requires the external watchdog and MLX cache-disabled worker wrapper.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from audit_pydantic_ai import audit_journal
from multikey_prose_pilot import digest, save
from replay_wide_mlx import ordinary_weights, replay_draw


def selected_rows(rows, plan):
    signature = lambda row: (row['key_slot'], row['case'], row['condition'])
    if (len(rows) != 128 or len(plan['schedule']) != 128
            or [signature(r) for r in rows] != [signature(r) for r in plan['schedule']]
            or [r['condition'] for r in rows[:2]] != ['ordinary', 'marked']
            or rows[0]['case'] != rows[1]['case']
            or rows[0]['key_slot'] != rows[1]['key_slot']):
        raise ValueError('Require the complete original schedule and its first pair')
    return rows[:2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (str(os.getppid()) != os.environ.get('KEYPRINT_WATCHDOG_PID')
            or 'KEYPRINT_MLX_MEMORY_LOG' not in os.environ):
        raise RuntimeError('External watchdog and guarded MLX wrapper required')
    import mlx.core as mx
    if mx.set_cache_limit(0) != 0:
        raise RuntimeError('MLX cache was not disabled before preflight')
    args.output.mkdir(mode=0o700)
    read = lambda path: json.loads(path.read_text())
    public, private = args.root/'public', args.root/'private'
    plan, identity, rows = read(public/'plan.json'), read(public/'identity.json'), read(private/'runs.json')
    selected = selected_rows(rows, plan)
    audit = read(public/'receipt-audit.json')
    if (audit['verified'] != 128 or audit['failures']
            or audit['runs_sha256'] != digest(private/'runs.json')
            or audit['plan_sha256'] != digest(public/'plan.json')):
        raise ValueError('Require unchanged complete receipt audit')
    import keyprint
    from keyprint.experimental.wide_mlx import verify_assets, NFCWideByteLevelBinding
    from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
    from keyprint._engine.legacy._impl.research.token_source_policy import TokenSourceSession
    package = Path(keyprint.__file__).parent
    sources = lambda: {str(f.relative_to(package)): digest(f) for f in package.rglob('*.py')}
    if sources() != plan['sdk_source_sha256'] or verify_assets(args.model) != identity['model_assets']:
        raise ValueError('Frozen SDK or model differs')
    for name, expected in identity['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Frozen runtime differs')
    keys = [(args.prior/f'private/key-{i}').read_bytes() for i in range(4)]
    if [hashlib.sha256(k).hexdigest() for k in keys] != plan['key_sha256']:
        raise ValueError('Frozen keys differ')
    scripts = [Path(__file__), *[Path(__file__).with_name(n) for n in (
        'replay_wide_mlx.py', 'audit_pydantic_ai.py', 'multikey_prose_pilot.py',
        'memory_watchdog.py', 'mlx_guarded_worker.py')]]
    commitments = {s.name: digest(s) for s in scripts}
    frozen = {'schema': 'keyprint.guarded-source-preflight.v1',
        'plan_sha256': digest(public/'plan.json'), 'runs_sha256': digest(private/'runs.json'),
        'scripts_sha256': commitments, 'rows': [r['review_id'] for r in selected],
        'selection': 'First original ordinary/marked pair, first eight steps each, fixed before execution',
        'steps_per_row': 8, 'purpose': 'Resource and exact prefix replay preflight, not full replay or quality acceptance',
        'cache_limit_bytes': 0, 'new_randomness': False, 'quality_acceptance': False}
    save(args.output/'plan.json', frozen)
    print('Frozen first-pair/eight-step preflight; loading pinned local model', flush=True)
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tokenizer = load(str(args.model), tokenizer_config={
        'trust_remote_code': False, 'local_files_only': True})
    if set(tokenizer.eos_token_ids) != {248046}:
        raise ValueError('Runtime EOS differs')
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),
        vocabulary_size=248320, special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
    if profile.digest != identity['profile_sha256']:
        raise ValueError('Profile differs')
    outcomes = []
    for row in selected:
        result = {'review_id': row['review_id'], 'matched_steps': 0, 'passed': False}
        outcomes.append(result)
        cache = make_prompt_cache(model)
        session = TokenSourceSession(profile, keys[row['key_slot']], condition=row['condition'])
        try:
            folder = private/row['review_id']
            for name, suffix in (('journal', '.jsonl'), ('report', '.json')):
                if digest(folder/(name+suffix)) != row[name+'_sha256']:
                    raise ValueError('Original receipt differs')
            events, report = audit_journal(folder/'journal.jsonl'), read(folder/'report.json')
            heads = [e for e in events if e['phase'] == 'prepared']
            draws = [e['draw'] for e in events if e['phase'] == 'commit_requested']
            tokens = report['committed_token_ids']
            if not len(heads) == len(draws) == len(tokens) or len(tokens) < 8:
                raise ValueError('Original step counts differ or insufficient prefix')
            inputs = events[0]['prompt_token_ids']
            for head, draw, token in zip(heads[:8], draws[:8], tokens[:8]):
                logits = model(mx.array([inputs]), cache=cache)[:, -1, :].astype(mx.float32)
                mx.eval(logits)
                raw = np.array(logits)
                if hashlib.sha256(raw.tobytes()).hexdigest() != head['raw_logits_sha256']:
                    raise ValueError('Native model-head bytes differ')
                base = ordinary_weights(raw, binding,
                    temperature=plan['settings']['temperature'], top_k=plan['settings']['top_k'])
                prepared = session.prepare(base)
                if hashlib.sha256(prepared.probabilities.tobytes()).hexdigest() != head['weights_sha256']:
                    raise ValueError('Reference weights differ')
                if replay_draw(prepared.probabilities, draw) != token:
                    raise ValueError('Recorded draw/token differs')
                session.commit(prepared, token)
                result['matched_steps'] += 1
                result.update(active_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory())
                save(args.output/'progress.json', outcomes)
                print(f"Prefix {len(outcomes)}/2 step {result['matched_steps']}/8 verified", flush=True)
                inputs = [token]
            result['passed'] = True
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
        finally:
            cache.clear(); session.close()
            mx.synchronize(); mx.clear_cache()
            save(args.output/'progress.json', outcomes)
    if sources() != plan['sdk_source_sha256'] or any(digest(s) != commitments[s.name] for s in scripts):
        raise ValueError('SDK or preflight implementation changed during execution')
    passed = all(r['passed'] and r['matched_steps'] == 8 for r in outcomes)
    save(args.output/'result.json', {'plan': frozen, 'passed': passed, 'rows': outcomes,
        'matched_steps': sum(r['matched_steps'] for r in outcomes),
        'full_replay': False, 'quality_acceptance': False, 'launch_ready': False})
    return 0 if passed else 1


if __name__ == '__main__': raise SystemExit(main())
