"""Bounded continuation ancestry, including planned pauses between full outputs.

CPU-only receipt validation. This supplements balanced_study.audit; it does not
replay native model heads, approve quality, or authorize automatic restarts.
"""
import json
from pathlib import Path
import random
import re
import shutil

from audit_paced_study import digest
from balanced_study import read, blind_packet, totals, audit
from balanced_inference import save
from balanced_continuation import inventory, verify_parent, audit_lineage, interrupted_tokens

HELPERS = ('balanced_chain.py', 'continue_balanced_chain.py')
ACCEPTANCE = dict(quality_acceptance=False, detector_calibrated=False, launch_ready=False)


def descriptors(path):
    data = read(path)
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError('Original study and first continuation required')
    stages = [{k: Path(r[k]).resolve() for k in ('root', 'supervisor', 'checkpoint')} for r in data]
    roots = [s['root'] for s in stages]
    if len(set(roots)) != len(roots):
        raise ValueError('Repeated ancestor directory')
    return stages


def bindings(stages):
    return [{k: digest(s[p] / f) if f else digest(s[p]) for k, p, f in (
        ('plan_sha256', 'root', 'public/plan.json'),
        ('runs_sha256', 'root', 'private/runs.json'),
        ('identity_sha256', 'root', 'public/identity.json'),
        ('supervisor_sha256', 'supervisor', 'result.json'),
        ('checkpoint_sha256', 'checkpoint', None))} for s in stages]


def verify_rows(root):
    """Check a schedule prefix without emitting conditions or detection scores."""
    plan = read(root / 'public/plan.json'); rows = read(root / 'private/runs.json')
    identity = read(root / 'public/identity.json'); prompts = read(root / 'private/prompt-ids.json')
    if len(rows) > 128 or len({r['review_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid prefix length or duplicate identity')
    for row, scheduled in zip(rows, plan['schedule'][:len(rows)], strict=True):
        if not re.fullmatch(r'[0-9a-f]{12}', row['review_id']) or any(row[k] != v for k, v in scheduled.items()):
            raise ValueError('Prefix assignment changed')
        folder = root / 'private' / row['review_id']; saved = read(folder / 'result.json')
        if (digest(folder / 'result.json') != row['result_sha256']
                or digest(folder / 'journal.jsonl') != row['journal_sha256']
                or any(row.get(k) != v for k, v in saved.items())
                or row['profile_sha256'] != identity['profile_sha256']
                or row['completion'] not in ('eos', 'cap', 'incomplete')
                or len(row['committed_token_ids']) > plan['settings']['max_tokens']
                or row['quality_acceptance'] is not False or row['detector_calibrated'] is not False):
            raise ValueError('Retained result changed')
        with (folder / 'journal.jsonl').open() as f:
            header = json.loads(next(f))
        expected = dict(phase='start', prompt_token_ids=prompts[row['case']],
                        profile_sha256=identity['profile_sha256'],
                        **{k: plan['settings'][k] for k in ('max_tokens', 'temperature', 'top_k')})
        if header != expected:
            raise ValueError('Retained prompt or settings changed')
        failed = any(k in row for k in ('error_type', 'decode_error', 'audit_error'))
        if not failed and (row.get('audit') != dict(verified_draws=len(row['committed_token_ids']),
                native_heads_replayed=False, quality_acceptance=False)
                or not row['committed_token_ids'] or row['completion'] not in ('eos', 'cap')
                or (row['completion'] == 'cap' and len(row['committed_token_ids']) != plan['settings']['max_tokens'])):
            raise ValueError('Completed result lacks sampler audit')
        scores = row['raw_counts']
        if len(scores) != 2:
            raise ValueError('Two token-path diagnostics required')
        for score in scores:
            if failed or row['completion'] != 'eos':
                if 'unavailable' not in score:
                    raise ValueError('Failure cannot become a clean negative')
            elif (score.get('token_path_only') is not True or score.get('detector_calibrated') is not False
                    or score.get('profile_sha256') != identity['profile_sha256']
                    or any(type(score[k]) is not int or score[k] < 0 for k in ('ones', 'events', 'trials'))
                    or score['ones'] > score['trials'] or score['events'] > len(row['committed_token_ids'])
                    or score['trials'] != score['events'] * 30):
                raise ValueError('Invalid retained diagnostic')
    return rows


def terminal_state(stage):
    root, guard_path, anchor_path = (stage[k] for k in ('root', 'supervisor', 'checkpoint'))
    guard = read(guard_path / 'result.json'); anchor = read(anchor_path)
    for key, path in [('plan_sha256', root / 'public/plan.json'), ('runs_sha256', root / 'private/runs.json'),
                      ('identity_sha256', root / 'public/identity.json'), ('supervisor_sha256', guard_path / 'result.json')]:
        if anchor[key] != digest(path):
            raise ValueError('Ancestor checkpoint changed')
    if guard.get('cleanup_verified') is not True or (root / 'public/results.json').exists():
        raise ValueError('Require cleaned, unfinished ancestor')
    rows = verify_rows(root); plan = read(root / 'public/plan.json')
    if len(rows) >= 128:
        raise ValueError('No remaining scheduled attempts')
    known = {r['review_id'] for r in rows}
    orphans = [p for p in (root / 'private').iterdir() if p.is_dir() and p.name not in known]
    if len(orphans) > 1:
        raise ValueError('More than one unfinished attempt')
    sealed = None
    if guard['status'] == 'completed' and guard['exit_code'] == 0:
        marker = read(root / 'public/batch-complete.json')
        spec = read(root / 'public' / plan['chain_link_file'])
        if (orphans or marker != dict(schema='keyprint.balanced-batch-pause.v1',
                runs_sha256=digest(root / 'private/runs.json'), next_index=len(rows),
                new_attempts=spec['max_new_attempts'], **ACCEPTANCE)
                or len(rows) != spec['first_unstarted_index'] + spec['max_new_attempts']):
            raise ValueError('Planned pause boundary changed')
    elif (guard['status'] == 'stopped' and guard['exit_code'] < 0
          and guard['reason'] in ('system_pressure', 'memory_budget', 'disk_headroom', 'time_budget')):
        if orphans:
            folder = orphans[0]; journal = folder / 'journal.jsonl'
            if (not re.fullmatch(r'[0-9a-f]{12}', folder.name)
                    or {p.name for p in folder.iterdir()} != {'journal.jsonl'}
                    or digest(journal) != anchor['unfinished_journal_sha256']):
                raise ValueError('Unfinished journal changed')
            attempt = plan['schedule'][len(rows)]
            start = dict(phase='start', prompt_token_ids=read(root / 'private/prompt-ids.json')[attempt['case']],
                         profile_sha256=read(root / 'public/identity.json')['profile_sha256'],
                         **{k: plan['settings'][k] for k in ('max_tokens', 'temperature', 'top_k')})
            tokens = interrupted_tokens(journal, start)
            result = dict(schema='keyprint.balanced-infrastructure-interruption.v1',
                condition=attempt['condition'], profile_sha256=start['profile_sha256'], completion='incomplete',
                committed_token_ids=tokens, text=None, error_type='InfrastructureInterrupted',
                error='Resource supervisor stopped worker; no retry or completed decoded output.',
                journal_sha256=digest(journal), interrupted_draws_audited=False,
                supervisor_sha256=digest(guard_path / 'result.json'), quality_acceptance=False, detector_calibrated=False)
            sealed = dict(attempt, review_id=folder.name, **result,
                raw_counts=[{'unavailable': 'Infrastructure interruption; not a clean negative'} for _ in range(2)])
    else:
        raise ValueError('Ancestor is not a verified stopped or planned-pause worker')
    return dict(rows=rows, sealed=sealed, files=inventory(root))


def audit_link(output, stages, state):
    plan = read(output / 'public/plan.json'); parent = stages[-1]['root']
    name = f'chain-link-{len(stages):03}.json'; spec = read(output / 'public' / name)
    expected_plan = dict(read(parent / 'public/plan.json'), chain_link_file=name,
                         chain_link_sha256=digest(output / 'public' / name))
    first = len(state['rows']) + int(state['sealed'] is not None)
    if (plan != expected_plan or spec['schema'] != 'keyprint.balanced-chain-link.v1'
            or spec['ancestors'] != bindings(stages) or spec['parent_files'] != state['files']
            or spec['first_unstarted_index'] != first or spec['preserved_outcomes'] != len(state['rows'])
            or spec['sealed_interrupted'] != int(state['sealed'] is not None)
            or type(spec['max_new_attempts']) is not int
            or not (spec['max_new_attempts'] == 0 if first == 128 else 1 <= spec['max_new_attempts'] <= 128 - first)
            or type(spec['model_load_authorized']) is not bool
            or any(spec[k] is not False for k in ACCEPTANCE)
            or spec['helpers_sha256'] != {n: digest(Path(__file__).with_name(n)) for n in HELPERS}):
        raise ValueError('Continuation chain binding changed')
    for name, sha in state['files'].items():
        if name == 'public/batch-complete.json':
            if digest(output / 'public' / f'parent-pause-{len(stages):03}.json') != sha:
                raise ValueError('Preserved pause receipt changed')
            continue
        if name not in ('public/plan.json', 'private/runs.json', 'public/batch-complete.json') and digest(output / name) != sha:
            raise ValueError('Preserved ancestor file changed')
    rows = verify_rows(output)
    if (rows[:len(state['rows'])] != state['rows'] or not first <= len(rows) <= first + spec['max_new_attempts']
            or (not spec['model_load_authorized'] and len(rows) != first)):
        raise ValueError('Preserved outcomes or bounded schedule changed')
    if state['sealed'] is not None:
        expected = dict(state['sealed']); folder = output / 'private' / expected['review_id']
        result = {k: v for k, v in expected.items() if k not in ('case', 'key_slot', 'review_id', 'raw_counts')}
        if read(folder / 'result.json') != result:
            raise ValueError('Interrupted outcome changed')
        expected['result_sha256'] = digest(folder / 'result.json')
        if rows[first - 1] != expected:
            raise ValueError('Interrupted outcome dropped or replaced')
    return dict(preserved_outcomes=len(state['rows']), sealed_interrupted=spec['sealed_interrupted'],
                new_attempts=len(rows) - first, first_unstarted_index=first, **ACCEPTANCE)


def verify_history(stages, bundle):
    if len(stages) < 2 or len({s['root'].resolve() for s in stages}) != len(stages):
        raise ValueError('Distinct original and continuation ancestors required')
    original = stages[0]
    verify_parent(original['root'], original['supervisor'], bundle, original['checkpoint'])
    audit_lineage(stages[1]['root'], original['root'], original['supervisor'], bundle, original['checkpoint'])
    if read(stages[1]['root'] / 'public/continuation.json')['new_model_load_authorized'] is not True:
        raise ValueError('Plan-only first continuation is not execution')
    state = terminal_state(stages[1])
    for index in range(2, len(stages)):
        audit_link(stages[index]['root'], stages[:index], state)
        state = terminal_state(stages[index])
    return state


def prepare_chain(stages, bundle, output, *, max_new_attempts, execute=False):
    state = verify_history(stages, bundle); first = len(state['rows']) + int(state['sealed'] is not None)
    if (type(max_new_attempts) is not int
            or not (max_new_attempts == 0 if first == 128 else 1 <= max_new_attempts <= 128 - first)):
        raise ValueError('Invalid planned batch length')
    output = output.resolve()
    if any(output == s['root'] or output.is_relative_to(s['root']) or s['root'].is_relative_to(output) for s in stages):
        raise ValueError('Output must be disjoint from all ancestors')
    parent = stages[-1]['root']; shutil.copytree(parent, output)
    # Preserve the old planned-pause marker in the immutable link inventory and
    # a separate byte-identical copy; it is not the new worker's completion.
    marker = output / 'public/batch-complete.json'
    if marker.exists():
        marker.rename(output / 'public' / f'parent-pause-{len(stages):03}.json')
    spec = dict(schema='keyprint.balanced-chain-link.v1', ancestors=bindings(stages), parent_files=state['files'],
        preserved_outcomes=len(state['rows']), sealed_interrupted=int(state['sealed'] is not None),
        first_unstarted_index=first, max_new_attempts=max_new_attempts, model_load_authorized=execute,
        helpers_sha256={n: digest(Path(__file__).with_name(n)) for n in HELPERS}, **ACCEPTANCE)
    name = f'chain-link-{len(stages):03}.json'; save(output / 'public' / name, spec)
    save(output / 'public/plan.json', dict(read(parent / 'public/plan.json'), chain_link_file=name,
                                          chain_link_sha256=digest(output / 'public' / name)))
    rows = list(state['rows'])
    if state['sealed'] is not None:
        row = dict(state['sealed']); folder = output / 'private' / row['review_id']
        save(folder / 'result.json', {k: v for k, v in row.items() if k not in ('case', 'key_slot', 'review_id', 'raw_counts')})
        row['result_sha256'] = digest(folder / 'result.json'); rows.append(row)
    save(output / 'private/runs.json', rows)
    if inventory(parent) != state['files']:
        raise ValueError('Parent changed during snapshot')
    audit_link(output, stages, state)
    return spec


def finish_batch(output, stages, bundle):
    state = verify_history(stages, bundle); lineage = audit_link(output, stages, state)
    plan = read(output / 'public/plan.json'); spec = read(output / 'public' / plan['chain_link_file'])
    rows = read(output / 'private/runs.json')
    known = {r['review_id'] for r in rows}
    if any(p.is_dir() and p.name not in known for p in (output / 'private').iterdir()):
        raise ValueError('Unindexed attempt cannot be hidden by finalization')
    if not spec['model_load_authorized'] or lineage['new_attempts'] != spec['max_new_attempts']:
        raise ValueError('Every planned new attempt must finish before recording a batch boundary')
    for name in ('public/batch-complete.json', 'public/results.json', 'private/blind-review.json'):
        if (output / name).exists():
            raise FileExistsError('Batch already finalized; do not overwrite')
    if len(rows) < 128:
        marker = dict(schema='keyprint.balanced-batch-pause.v1', runs_sha256=digest(output / 'private/runs.json'),
                      next_index=len(rows), new_attempts=lineage['new_attempts'], **ACCEPTANCE)
        save(output / 'public/batch-complete.json', marker)
        return marker
    review = blind_packet(rows, read(output / 'private/cases.json'), read(output / 'private/rubrics.json'))
    random.SystemRandom().shuffle(review); save(output / 'private/blind-review.json', review)
    result = dict(schema='keyprint.balanced-evaluation-results.v1', **totals(rows),
        plan_sha256=digest(output / 'public/plan.json'), runs_sha256=digest(output / 'private/runs.json'),
        identity_sha256=digest(output / 'public/identity.json'), review_sha256=digest(output / 'private/blind-review.json'),
        continuation=lineage, **ACCEPTANCE)
    save(output / 'public/results.json', result)
    save(output / 'public/cohort-audit.json', dict(audit(output, bundle), continuation=lineage))
    return result
