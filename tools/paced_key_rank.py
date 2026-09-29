"""Opened-cohort random-key rank diagnostic; no inference or calibration claim.

Fresh decoys and one score/cutoff are frozen before their scores are evaluated.
Ranks condition on recorded tokenizer paths. They are not arbitrary-text detector
p-values, fixed-key FPR estimates, or independent trials across reused tasks/keys.
"""
import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import statistics

from audit_paced_study import audit, digest
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import pack

DECOYS = 199


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def compile_paths(profile, paths):
    """Compile exactly replay_events eligibility, retaining shared-message reuse."""
    unique, messages, indexes = {}, [], []
    for tokens in paths:
        if len(tokens) > profile.config.max_steps:
            raise ValueError('Path exceeds profile cap')
        used, context, selected = set(), (), []
        for token in tokens:
            label = profile.label(token)
            if label is None:
                continue
            if context not in used:
                signature = (context, label)
                if signature not in unique:
                    unique[signature] = len(messages)
                    messages.append(tuple(pack([profile._domain, pack(context),
                        layer.to_bytes(4, 'big'), label])
                        for layer in range(profile.config.layers)))
                selected.append(unique[signature])
            used.add(context)
            context = (*context, label)[-profile.config.history:]
        indexes.append(selected)
    return messages, indexes


def score_paths(key, messages, indexes):
    if type(key) is not bytes or len(key) != 32:
        raise ValueError('Exactly 32 key bytes required')
    template = hmac.new(key, digestmod='sha256')
    counts = []
    for event in messages:
        count = 0
        for message in event:
            state = template.copy()
            state.update(message)
            count += state.digest()[0] & 1
        counts.append(count)
    return [sum(counts[i] for i in selected) for selected in indexes]


def key_rank(observed, decoys, trials):
    if (type(trials) is not int or trials < 1 or not decoys
            or any(type(v) is not int or not 0 <= v <= trials
                   for v in [observed, *decoys])):
        raise ValueError('Available integer scores and positive trials required')
    numerator = 1 + sum(v >= observed for v in decoys)
    denominator = len(decoys) + 1
    return {'rank_numerator': numerator, 'rank_denominator': denominator,
            'random_key_rank': numerator / denominator,
            'at_or_below_one_percent': 100 * numerator <= denominator}


def summarize(rows):
    if (len(rows) != 128 or len({r['review_id'] for r in rows}) != 128
            or len({r['case'] for r in rows}) != 16):
        raise ValueError('Complete 128-attempt cohort required')
    signatures = {(r['case'], r['key_slot'], r['condition']) for r in rows}
    expected = {(c, k, a) for c in {r['case'] for r in rows}
                for k in range(4) for a in ('ordinary', 'marked')}
    if signatures != expected:
        raise ValueError('Missing or duplicate scheduled attempt')
    groups = {}
    for arm in ('ordinary', 'marked'):
        selected = [r for r in rows if r['condition'] == arm]
        available = [r for r in selected if 'rank' in r]
        groups[arm] = {'attempts': len(selected), 'available': len(available),
            'unavailable': len(selected) - len(available),
            'at_or_below_one_percent': sum(r['rank']['at_or_below_one_percent'] for r in available),
            'median_random_key_rank': statistics.median(r['rank']['random_key_rank'] for r in available)
                if available else None}
    return {'groups': groups, 'detector_calibrated': False,
            'quality_acceptance': False, 'launch_ready': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'original', 'prior', 'model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if str(os.getppid()) != os.environ.get('KEYPRINT_WATCHDOG_PID'):
        raise RuntimeError('External memory watchdog required even for offline scoring')
    checked = audit(args.study, args.original)
    if checked['audit_errors']:
        raise ValueError('Resolve original audit errors first')
    study_plan = json.loads((args.study/'public/plan.json').read_text())
    identity = json.loads((args.study/'public/identity.json').read_text())
    rows = json.loads((args.study/'private/runs.json').read_text())
    keys = [(args.prior/f'private/key-{i}').read_bytes() for i in range(4)]
    if (any(len(k) != 32 for k in keys)
            or [hashlib.sha256(k).hexdigest() for k in keys] != study_plan['key_sha256']):
        raise ValueError('Generation keys differ')
    # Only tokenizer assets are needed. No weight file is read or model imported.
    assets = {n: digest(args.model/n) for n in ('tokenizer.json', 'tokenizer_config.json')}
    if any(h != study_plan['model_assets'][n] for n, h in assets.items()):
        raise ValueError('Tokenizer assets changed')
    for name in ('transformers', 'tokenizers', 'numpy'):
        if importlib.metadata.version(name) != study_plan['dependencies'][name]:
            raise ValueError('Tokenizer runtime changed')
    import keyprint
    package = Path(keyprint.__file__).parent
    if {str(f.relative_to(package)): digest(f) for f in package.rglob('*.py')} != study_plan['sdk_source_sha256']:
        raise ValueError('Frozen SDK changed')
    args.output.mkdir(mode=0o700)
    private, public = args.output/'private', args.output/'public'
    private.mkdir(mode=0o700); public.mkdir()
    controls = [os.urandom(32) for _ in range(DECOYS)]
    if len(set(controls + keys)) != DECOYS + 4:
        raise RuntimeError('Key collision; retain failed attempt, do not redraw')
    for i, key in enumerate(controls):
        with (private/f'decoy-{i}.key').open('xb') as f:
            f.write(key)
    plan = {'schema': 'keyprint.paced-key-rank-plan.v1',
        'script_sha256': digest(Path(__file__)), 'cohort': checked,
        'tokenizer_assets': assets, 'profile_sha256': identity['profile_sha256'],
        'decoy_key_sha256': [hashlib.sha256(k).hexdigest() for k in controls],
        'decoys': DECOYS, 'score': 'Unweighted ones across all 30 replay-eligible layers/events',
        'rank': '(1 + count(decoy score >= owner score)) / 200; conservative ties',
        'cutoff': 'rank <= 0.01, integer comparison; no score or cutoff search',
        'selection': 'All 128 existing paths, original order; unavailable rows retained',
        'scope': 'Opened development cohort; fresh keys but reused outputs, tasks and generation keys. Token-path diagnostic only, not calibrated deployment detection or a p-value.',
        'model_inference': False, 'quality_acceptance': False, 'detector_calibrated': False}
    write(public/'plan.json', plan)
    from transformers import AutoTokenizer
    from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding
    from paced_source_session import PacedProfile, Config
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False, local_files_only=True)
    binding = NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),
        vocabulary_size=248320, special_ids=tokenizer.all_special_ids, eos_ids=[248046])
    profile = PacedProfile(binding.pieces, tokenizer_identity=binding.digest,
        eos_ids=binding.eos_ids, config=Config(max_steps=1024))
    if binding.digest != identity['binding_sha256'] or profile.digest != identity['profile_sha256']:
        raise ValueError('Tokenizer/profile reconstruction differs')
    messages, indexes = compile_paths(profile, [r['committed_token_ids'] for r in rows])
    owner_scores = [score_paths(k, messages, indexes) for k in keys]
    for i, row in enumerate(rows):
        for offset, original in enumerate(row['raw_counts']):
            slot = (row['key_slot'] + offset) % 4
            if (owner_scores[slot][i] != original['ones']
                    or len(indexes[i]) != original['events']
                    or len(indexes[i]) * 30 != original['trials']):
                raise ValueError('Compiled scores differ from original independent replay')
    write(public/'reconciliation.json', {'outputs': 128, 'original_scores_matched': 256,
        'unique_events': len(messages), 'model_loaded': False})
    null_scores = []
    with (private/'decoy-scores.jsonl').open('x') as f:
        for i, key in enumerate(controls):
            values = score_paths(key, messages, indexes)
            f.write(json.dumps({'index': i, 'ones': values})+'\n'); f.flush()
            null_scores.append(values)
            if (i + 1) % 10 == 0:
                print(json.dumps({'completed_decoys': i + 1, 'planned_decoys': DECOYS}), flush=True)
    results = []
    for i, row in enumerate(rows):
        item = {k: row[k] for k in ('review_id', 'case', 'condition', 'key_slot')}
        trials = len(indexes[i]) * 30
        if trials:
            item.update(ones=owner_scores[row['key_slot']][i], trials=trials,
                rank=key_rank(owner_scores[row['key_slot']][i], [r[i] for r in null_scores], trials))
        else:
            item['unavailable'] = 'No eligible events'
        results.append(item)
    if audit(args.study, args.original) != checked or digest(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Inputs or diagnostic code changed during run')
    result = summarize(results)
    result.update(plan_sha256=digest(public/'plan.json'),
        decoy_scores_sha256=digest(private/'decoy-scores.jsonl'), rows=results, scope=plan['scope'])
    write(public/'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
