"""Audit prompt-aware null records using independent graph and score replay.

Checks source selection, same confirmation keys, literal tokens, original prompt,
every retained probability/score term and aggregate counts. Does not rerun model
kernels or turn a corpus screen into a population false-positive guarantee.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from scipy.stats import beta

from audit_prompt_confirmation import validate_groups, check_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_prefix(path):
    raw = path.read_bytes()
    return raw[:raw.rfind(b'\n') + 1]


def records(path, finished, *, follow=False, timeout=7200):
    """Follow append-only complete lines; never accept a changed audited prefix."""
    start = time.monotonic(); consumed = b''
    while True:
        raw = record_prefix(path)
        if not raw.startswith(consumed):
            raise ValueError('Retained result prefix changed during audit')
        for line in raw[len(consumed):].splitlines(keepends=True):
            value = json.loads(line)
            consumed += line
            yield value
        if not follow or finished.exists():
            # Completion can race the first snapshot. Read the final bytes again.
            if follow and record_prefix(path) != consumed:
                continue
            return
        if time.monotonic() - start >= timeout:
            raise TimeoutError('Null run has not produced a terminal summary')
        time.sleep(2)


def verify_summary(rows, summary, planned=500):
    available = [row for row in rows if 'error' not in row]
    hits = sum(row['flagged'] for row in available)
    complete = len(rows) == planned and len(available) == planned and summary['fatal'] is None
    upper = (1. if hits == planned else float(beta.ppf(.975, hits + 1, planned - hits))) if complete else None
    expected = {'status': 'completed' if complete else 'incomplete', 'planned': planned,
        'attempts': len(rows), 'available': len(available), 'unavailable': planned - len(available),
        'errors': len(rows) - len(available), 'false_hits': hits,
        'observed_rate': hits / len(available) if available else None, 'target': .01,
        'null_screen_passed': bool(complete and upper <= .01),
        'deployment_calibrated': False, 'quality_acceptance': False, 'prompt_free': False}
    assert all(summary[k] == v for k, v in expected.items())
    if upper is None:
        assert summary['iid_only_upper_97_5_percent'] is None
    else:
        assert abs(summary['iid_only_upper_97_5_percent'] - upper) < 1e-12
    return {'complete': complete, 'false_hits': hits, 'iid_only_upper_97_5_percent': upper}


def main():
    if not __debug__:
        raise RuntimeError('Evidence audit requires Python assertions enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'model'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--partial-output', type=Path, help='Audit completed record prefix only; never marks study complete')
    parser.add_argument('--follow', action='store_true', help='Audit appended completed records while local inference continues')
    args = parser.parse_args(); public = args.study / 'public'
    if args.follow and args.partial_output:
        parser.error('--follow and --partial-output are mutually exclusive')
    audit_digest = sha(Path(__file__))
    plan = json.loads((public / 'plan.json').read_text())
    assert sha(args.source) == plan['source_sha256']
    assert sha(Path(__file__).with_name('validate_prompt_null.py')) == plan['script_sha256']
    for name, digest in plan['dependencies_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    from prompt_confirmation_selection import source_indices
    from prompt_null_selection import select, prompt, SALT
    previous = set()
    for name, digest in plan['prior_manifests_sha256'].items():
        assert sha(Path(name)) == digest
        previous.update(source_indices(json.loads(Path(name).read_text())))
    assert previous == set(plan['excluded_source_indices'])
    source = [json.loads(line) for line in args.source.read_bytes().splitlines()]
    selected, selection = select(source, previous)
    assert selected == plan['tasks'] and selection == plan['selection'] and plan['selection_salt'] == SALT
    assert plan['planned_documents'] == 500 and len(selected) == 500
    group_audit = validate_groups(source, {**plan, 'tasks': [dict(task, prompt=prompt(source[task['source_index']])) for task in selected]})
    cp = Path(plan['confirmation']) / 'public'
    assert str((cp / 'plan.json').resolve()) in plan['prior_manifests_sha256']
    for name, digest in plan['confirmation_sha256'].items():
        assert sha(cp / name) == digest
    old_plan = json.loads((cp / 'plan.json').read_text())
    keys = [(cp.parent / f'owner-{i}.key').read_bytes() for i in range(2)]
    assert len(set(keys)) == 2 and all(len(k) == 32 for k in keys)
    assert [hashlib.sha256(k).hexdigest() for k in keys] == plan['key_commitments'] == old_plan['key_commitments']
    assert plan['fixed_log_cutoff'] == math.log(200.) and plan['mixture'] == .5
    assert plan['chunk_size'] == 64 and plan['temperature'] == .7 and plan['top_k'] == 100 and plan['target'] == .01
    from keyprint.backends.mlx import verify_assets, ASSETS
    from keyprint import sampling
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from transformers import AutoTokenizer
    verify_assets(args.model); binding = runtime_binding(max_steps=2048)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    identity = json.loads((public / 'identity.json').read_text())
    assert identity == {'model_assets': ASSETS, 'profile': binding.profile.identity_receipt(), 'sampling': sampling.identity()}
    old_identity = json.loads((cp / 'identity.json').read_text())
    assert all(old_identity[k] == v for k, v in identity.items())
    rows = []
    totals = {'records': 0, 'available': 0, 'heads': 0, 'scores': 0, 'terms': 0, 'transforms': 0}
    maximum_term_error = maximum_score_error = 0.
    for index, row in enumerate(records(public / 'results.jsonl', public / 'summary.json', follow=args.follow)):
        assert index < 500
        task = selected[index]
        rows.append(row); totals['records'] += 1
        name = f'null-{index:03d}'
        assert row['id'] == name and all(row[k] == v for k, v in task.items())
        original = source[task['source_index']]
        assert task['text_sha256'] == hashlib.sha256(original['response'].encode()).hexdigest()
        assert task['words'] == len(original['response'].split()) and 100 <= task['words'] <= 400
        if 'error' in row:
            assert 'flagged' not in row
            continue
        head_path = args.study / (name + '.heads.json'); score_path = args.study / (name + '.scores.json')
        assert sha(head_path) == row['heads_sha256'] and sha(score_path) == row['scores_sha256']
        heads = json.loads(head_path.read_text()); scores = json.loads(score_path.read_text())
        assert len(scores) == 2 and heads['token_ids'] == list(binding.encode_visible(original['response']))
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt(original)}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert heads['conditioning_prefix_ids'] == prefix and heads['text_sha256'] == task['text_sha256']
        assert heads['original_prompt_sha256'] == task['prompt_sha256'] and heads['original_prompt_used']
        assert not heads['private_generation_data_used'] and not heads['key_used_for_model_inference']
        assert row['tokens'] == len(heads['token_ids']) == len(heads['heads'])
        for key, value in zip(keys, scores, strict=True):
            checked = check_score(binding.profile, key, heads, value)
            totals['terms'] += checked['terms']; totals['transforms'] += checked['transforms']; totals['scores'] += 1
            maximum_term_error = max(maximum_term_error, checked['term_error'])
            maximum_score_error = max(maximum_score_error, checked['score_error'])
        assert row['working_log_ratios'] == [v['working_log_ratio'] for v in scores]
        assert row['flags'] == [v['flagged'] for v in scores] and row['flagged'] == any(row['flags'])
        assert row['scored_events'] == scores[0]['scored_events'] and row['outside_support'] == scores[0]['outside_support']
        totals['available'] += 1; totals['heads'] += len(heads['heads'])
        if (index + 1) % 25 == 0:
            print(json.dumps({'audited': index + 1, 'totals': totals}), flush=True)
    assert rows and sha(Path(__file__)) == audit_digest
    raw = record_prefix(public / 'results.jsonl')
    # A partial audit binds only its captured prefix, even if inference appended
    # more rows while those scores were checked.
    if args.partial_output:
        raw = b''.join(raw.splitlines(keepends=True)[:len(rows)])
    else:
        assert raw == (public / 'results.jsonl').read_bytes()
    assert [json.loads(line) for line in raw.splitlines()] == rows
    result = {'status': 'partial_integrity_pass' if args.partial_output else 'pass', 'scope': __doc__,
        'group_audit': group_audit, 'totals': totals,
        'maximum_term_error': maximum_term_error, 'maximum_score_error': maximum_score_error,
        'plan_sha256': sha(public / 'plan.json'), 'results_prefix_sha256': hashlib.sha256(raw).hexdigest(),
        'results_prefix_bytes': len(raw), 'audit_source_sha256': audit_digest,
        'helpers_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('audit_prompt_confirmation.py', 'audit_prompt_conditioned_likelihood.py')},
        'model_kernels_rerun': False, 'deployment_calibrated': False, 'sdk_promotion': False}
    if not args.partial_output:
        summary = json.loads((public / 'summary.json').read_text())
        result['summary_audit'] = verify_summary(rows, summary)
        result['summary_sha256'] = sha(public / 'summary.json')
    with (args.partial_output or public / 'integrity.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
