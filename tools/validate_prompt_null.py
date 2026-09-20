"""Fresh prompt-aware 500-document null screen with the confirmation keys.

Run actual Qwen predictive inference on unchanged source human responses, then
score both frozen keys. One document is one maximum-across-keys observation.
This is not prompt-free detection, a quality test or deployment calibration.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

from prompt_confirmation_selection import SOURCE_SHA256, history, sha
from prompt_null_selection import select, prompt, SALT, DOCUMENTS
from prompt_conditioned_likelihood import measure
from surrogate_likelihood import score, LOG_CUTOFF, MIXTURE, CHUNK_SIZE
from validate_null_corpus import iid_binomial_upper
from develop_surrogate_likelihood import write

TARGET = .01
DISK_FLOOR = 2 * 1024**3


def summarize(rows, fatal=None):
    available = [row for row in rows if 'error' not in row]
    hits = sum(row['flagged'] for row in available)
    complete = len(rows) == DOCUMENTS and len(available) == DOCUMENTS and fatal is None
    upper = iid_binomial_upper(hits, DOCUMENTS) if complete else None
    return {'status': 'completed' if complete else 'incomplete', 'planned': DOCUMENTS,
        'attempts': len(rows), 'available': len(available), 'unavailable': DOCUMENTS - len(available),
        'errors': len(rows) - len(available), 'fatal': fatal, 'false_hits': hits,
        'observed_rate': hits / len(available) if available else None,
        'iid_only_upper_97_5_percent': upper, 'target': TARGET,
        'null_screen_passed': bool(complete and upper <= TARGET),
        'deployment_calibrated': False, 'prompt_free': False, 'quality_acceptance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'history-root', 'confirmation', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if sha(raw) != SOURCE_SHA256:
        raise ValueError('Pinned source differs')
    source = [json.loads(line) for line in raw.splitlines()]
    cp = args.confirmation / 'public'
    prior = json.loads((cp / 'plan.json').read_text())
    audit = json.loads((cp / 'integrity.json').read_text())
    previous = json.loads((cp / 'summary.json').read_text())
    if (audit['status'] != 'pass' or not previous['confirmation_screen_passed']
            or audit['plan_sha256'] != sha((cp / 'plan.json').read_bytes())
            or audit['summary_sha256'] != sha((cp / 'summary.json').read_bytes())):
        raise ValueError('Audited complete confirmation required')
    for name, digest in prior['dependencies_sha256'].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError('Frozen confirmation dependency differs')
    if prior['fixed_log_cutoff'] != LOG_CUTOFF or prior['mixture'] != MIXTURE or prior['chunk_size'] != CHUNK_SIZE:
        raise ValueError('Frozen score settings differ')
    keys = [(args.confirmation / f'owner-{i}.key').read_bytes() for i in range(2)]
    if len(set(keys)) != 2 or any(len(k) != 32 for k in keys) or [sha(k) for k in keys] != prior['key_commitments']:
        raise ValueError('Confirmation keys differ')
    manifests, excluded = history(args.history_root)
    if str((cp / 'plan.json').resolve()) not in manifests:
        raise ValueError('Fresh confirmation must be excluded from null selection')
    tasks, selection = select(source, excluded)
    if shutil.disk_usage(args.output.parent).free < DISK_FLOOR:
        raise OSError('Less than 2 GiB free before null study')
    args.output.mkdir(mode=0o700)
    public = args.output / 'public'; public.mkdir()
    deps = ('prompt_null_selection.py', 'prompt_confirmation_selection.py', 'prompt_conditioned_likelihood.py',
            'surrogate_likelihood.py', 'predictability_filter.py', 'log_tournament.py', 'validate_null_corpus.py',
            'develop_surrogate_likelihood.py')
    plan = {'scope': __doc__, 'source': prior['source'], 'revision': prior['revision'],
        'attribution': prior['attribution'], 'license': prior['license'], 'source_sha256': SOURCE_SHA256,
        'script_sha256': sha(Path(__file__).read_bytes()),
        'dependencies_sha256': {n: sha(Path(__file__).with_name(n).read_bytes()) for n in deps},
        'confirmation': str(args.confirmation.resolve()),
        'confirmation_sha256': {n: sha((cp / n).read_bytes()) for n in ('plan.json', 'summary.json', 'integrity.json', 'identity.json')},
        'prior_manifests_sha256': manifests, 'excluded_source_indices': sorted(excluded),
        'selection': selection, 'selection_salt': SALT, 'tasks': tasks,
        'key_commitments': prior['key_commitments'], 'key_rule': 'Same two keys as fresh paired confirmation; neither key refit',
        'temperature': .7, 'top_k': 100, 'chunk_size': CHUNK_SIZE, 'mixture': MIXTURE,
        'fixed_log_cutoff': LOG_CUTOFF, 'planned_documents': DOCUMENTS, 'target': TARGET,
        'decision': 'Flag document if either key score >= log(200); frozen threshold, no fitting',
        'acceptance': 'All 500 available and one-sided 97.5% IID-only binomial upper bound <= 0.01',
        'unit': 'One document across two keys; never 1000 independent observations',
        'conditioning': 'Original instruction and reference context with unchanged source human response',
        'failure_rule': 'Retain all attempts; no retries, replacements or truncation; unavailable never counts negative',
        'disk_floor_bytes': DISK_FLOOR,
        'uncertainty': 'English Dolly answers; near duplicates, authors and topics may violate IID. Model training overlap unknown. No fixed-key population or deployment guarantee.'}
    write(public / 'plan.json', plan)
    print(json.dumps({'stage': 'frozen', 'selection': selection}), flush=True)
    rows = []; fatal = None; started = time.monotonic()
    try:
        from keyprint.backends.mlx import MLXModel, ASSETS
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        from keyprint import sampling
        binding = runtime_binding(max_steps=2048)
        backend = MLXModel.load(args.model)
        identity = {'model_assets': ASSETS, 'profile': binding.profile.identity_receipt(), 'sampling': sampling.identity()}
        prior_identity = json.loads((cp / 'identity.json').read_text())
        if any(prior_identity[k] != v for k, v in identity.items()):
            raise ValueError('Confirmation model/profile/sampling identity differs')
        write(public / 'identity.json', identity)
        with (public / 'results.jsonl').open('x') as stream:
            for index, task in enumerate(tasks):
                name = f'null-{index:03d}'
                row = {'id': name, **task}; begin = time.monotonic()
                try:
                    if shutil.disk_usage(args.output).free < DISK_FLOOR:
                        raise OSError('Less than 2 GiB free; sample unavailable without retry')
                    item = source[task['source_index']]
                    heads = measure(backend, binding, item['response'], prompt(item))
                    head_path = args.output / (name + '.heads.json'); write(head_path, heads)
                    values = [score(binding.profile, key, heads) for key in keys]
                    score_path = args.output / (name + '.scores.json'); write(score_path, values)
                    row.update(working_log_ratios=[v['working_log_ratio'] for v in values],
                        flags=[v['flagged'] for v in values], flagged=any(v['flagged'] for v in values),
                        scored_events=values[0]['scored_events'], outside_support=values[0]['outside_support'],
                        tokens=len(heads['token_ids']), heads_sha256=sha(head_path.read_bytes()), scores_sha256=sha(score_path.read_bytes()))
                except Exception as exc:
                    row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                row['seconds'] = time.monotonic() - begin
                rows.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                print(json.dumps({'completed': len(rows), 'id': name, 'flagged': row.get('flagged'),
                    'error': row.get('error'), 'seconds': row['seconds']}), flush=True)
    except Exception as exc:
        fatal = {'type': type(exc).__name__, 'message': str(exc)}
    summary = summarize(rows, fatal); summary['seconds'] = time.monotonic() - started
    write(public / 'summary.json', summary); print(json.dumps(summary), flush=True)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
