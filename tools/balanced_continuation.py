"""Preserve a stopped study; seal its unfinished attempt without regenerating it.

Lineage validation is additional to balanced_study.audit. Original artifacts
remain untouched. Receipt reconciliation is not independent native-head replay.
"""
import json
from pathlib import Path
import re
import shutil

from audit_paced_study import digest
from balanced_inference import save
from balanced_study import read, blind_packet, totals, audit
from freeze_balanced_evaluation import validate

HELPERS=('balanced_continuation.py','continue_balanced_evaluation.py')


def inventory(root):
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Symlink in study snapshot')
        if path.is_file():result[str(path.relative_to(root))]=digest(path)
    return result


def interrupted_tokens(path,expected_start):
    """Validate event order only; do not claim interrupted draws were audited."""
    state='start';tokens=[];draw=None
    with path.open() as f:
        for line in f:
            event=json.loads(line)
            if state=='start':
                if event!=expected_start:raise ValueError('Interrupted prompt/profile/settings differ')
                state='prepared';continue
            phase=event['phase']
            if phase!=state or event.get('index')!=len(tokens):
                raise ValueError('Invalid interrupted journal order')
            if phase=='prepared':state='drawn'
            elif phase=='drawn':draw=event['draw']['token_index'];state='committed'
            elif phase=='committed':
                token=event['token_id']
                if type(token) is not int or token<0 or token!=draw:raise ValueError('Interrupted commit differs from draw')
                tokens.append(token);state='prepared'
            else:raise ValueError('Unexpected interrupted journal event')
    if state=='start' or len(tokens)>=expected_start['max_tokens']:
        raise ValueError('Not an unfinished bounded attempt')
    return tokens


def verify_parent(root,supervisor,bundle,checkpoint):
    manifest=validate(bundle);plan=read(root/'public/plan.json');rows=read(root/'private/runs.json')
    identity=read(root/'public/identity.json');guard=read(supervisor/'result.json');anchor=read(checkpoint)
    if (guard.get('status')!='stopped' or guard.get('reason') not in
            ('system_pressure','memory_budget','disk_headroom','time_budget')
            or guard.get('cleanup_verified') is not True or guard.get('exit_code',0)>=0
            or (root/'public/results.json').exists() or 'continuation_sha256' in plan):
        raise ValueError('Require stopped, cleaned original study without final results')
    for key,path in [('plan_sha256',root/'public/plan.json'),('runs_sha256',root/'private/runs.json'),
                     ('identity_sha256',root/'public/identity.json'),('supervisor_sha256',supervisor/'result.json')]:
        if anchor[key]!=digest(path):raise ValueError('Interrupted checkpoint binding differs')
    if (anchor['schema']!='keyprint.balanced-interrupted-study.v1' or not 0<len(rows)<128
            or anchor['completed_eos']!=len(rows) or anchor['scheduled']!=128
            or anchor['interrupted_attempts']!=1 or anchor['unstarted_attempts']!=127-len(rows)
            or any(anchor[k] is not False for k in ('quality_acceptance','detector_calibrated','launch_ready'))
            or plan['schema']!='keyprint.balanced-evaluation-run.v1'
            or plan['model_load_authorized_in_this_invocation'] is not True
            or plan['schedule']!=manifest['schedule'] or plan['policy']!=manifest['policy']
            or plan['bundle_manifest_sha256']!=digest(bundle/'public/manifest.json')
            or plan['bundle_protocol_sha256']!=digest(bundle/'public/protocol.json')
            or plan['settings']!=read(bundle/'public/protocol.json')['settings']
            or plan['preflight']['profile_sha256']!=identity['profile_sha256']):
        raise ValueError('Interrupted study scope differs')
    from run_balanced_evaluation import HELPERS as ORIGINAL_HELPERS
    if set(plan['scripts_sha256'])!=set(ORIGINAL_HELPERS):raise ValueError('Missing original helper binding')
    for name,sha in plan['scripts_sha256'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('Original helper changed')
    for name in ('cases.json','rubrics.json'):
        if read(root/'private'/name)!=read(bundle/'private'/name):raise ValueError('Original source inputs differ')
    prompts=read(root/'private/prompt-ids.json');seen=set()
    for row,scheduled in zip(rows,manifest['schedule'][:len(rows)],strict=True):
        uid=row['review_id']
        if not re.fullmatch(r'[0-9a-f]{12}',uid) or uid in seen or any(row[k]!=v for k,v in scheduled.items()):
            raise ValueError('Original prefix order or identity differs')
        seen.add(uid);folder=root/'private'/uid;saved=read(folder/'result.json')
        if (digest(folder/'result.json')!=row['result_sha256'] or digest(folder/'journal.jsonl')!=row['journal_sha256']
                or any(row.get(k)!=v for k,v in saved.items())
                or row['completion']!='eos' or row['profile_sha256']!=identity['profile_sha256']
                or any(k in row for k in ('error_type','decode_error','audit_error'))
                or not row['committed_token_ids']
                or row['audit']!=dict(verified_draws=len(row['committed_token_ids']),native_heads_replayed=False,quality_acceptance=False)):
            raise ValueError('Original completed output differs')
        with (folder/'journal.jsonl').open() as f:header=json.loads(next(f))
        if header['prompt_token_ids']!=prompts[row['case']]:raise ValueError('Original prompt tokens differ')
    orphans=[p for p in (root/'private').iterdir() if p.is_dir() and p.name not in seen]
    if len(orphans)!=1 or not re.fullmatch(r'[0-9a-f]{12}',orphans[0].name):
        raise ValueError('Exactly one unfinished attempt required')
    folder=orphans[0];journal=folder/'journal.jsonl'
    if (set(p.name for p in folder.iterdir())!={'journal.jsonl'}
            or digest(journal)!=anchor['unfinished_journal_sha256']):
        raise ValueError('Unfinished attempt changed or has a result')
    attempt=plan['schedule'][len(rows)]
    start=dict(phase='start',prompt_token_ids=prompts[attempt['case']],profile_sha256=identity['profile_sha256'],
               **{k:plan['settings'][k] for k in ('max_tokens','temperature','top_k')})
    tokens=interrupted_tokens(journal,start)
    if (anchor['completed_committed_tokens']!=sum(len(r['committed_token_ids']) for r in rows)
            or anchor.get('unfinished_committed_journal_entries',len(tokens))!=len(tokens)):
        raise ValueError('Interrupted token counts differ')
    sealed=dict(schema='keyprint.balanced-infrastructure-interruption.v1',condition=attempt['condition'],
        profile_sha256=identity['profile_sha256'],completion='incomplete',committed_token_ids=tokens,text=None,
        error_type='InfrastructureInterrupted',error='Memory/resource supervisor stopped original worker; no retry or completed decoded output.',
        journal_sha256=digest(journal),interrupted_draws_audited=False,
        supervisor_sha256=digest(supervisor/'result.json'),quality_acceptance=False,detector_calibrated=False)
    interrupted=dict(attempt,review_id=folder.name,**sealed,
        raw_counts=[{'unavailable':'Infrastructure interruption; not a clean negative'} for _ in range(2)])
    return dict(plan=plan,rows=rows,interrupted=interrupted,sealed=sealed,parent_files=inventory(root),
                checkpoint_sha256=digest(checkpoint),supervisor_sha256=digest(supervisor/'result.json'))


def prepare(root,supervisor,bundle,checkpoint,output,*,execute=False):
    parent=verify_parent(root,supervisor,bundle,checkpoint)
    spec=dict(schema='keyprint.balanced-continuation.v1',new_model_load_authorized=execute,parent_files=parent['parent_files'],
        checkpoint_sha256=parent['checkpoint_sha256'],supervisor_sha256=parent['supervisor_sha256'],
        preserved_completed=len(parent['rows']),sealed_interruption_id=parent['interrupted']['review_id'],
        first_unstarted_index=len(parent['rows'])+1,planned_new_attempts=127-len(parent['rows']),
        helpers_sha256={n:digest(Path(__file__).with_name(n)) for n in HELPERS},
        policy='Preserve original outputs; record infrastructure interruption as failure; run only previously unstarted rows once.',
        quality_acceptance=False,detector_calibrated=False,launch_ready=False)
    if output.exists():raise FileExistsError(output)
    shutil.copytree(root,output);output.chmod(0o700)
    # Original remains untouched; only the new lineage adds an explicit error receipt.
    save(output/'public/parent-checkpoint.json',read(checkpoint))
    save(output/'public/parent-supervisor.json',read(supervisor/'result.json'))
    save(output/'public/continuation.json',spec)
    save(output/'public/plan.json',dict(parent['plan'],continuation_sha256=digest(output/'public/continuation.json')))
    row=parent['interrupted'];folder=output/'private'/row['review_id'];save(folder/'result.json',parent['sealed'])
    row['result_sha256']=digest(folder/'result.json')
    save(output/'private/runs.json',parent['rows']+[row])
    if inventory(root)!=parent['parent_files']:raise ValueError('Original changed during copy')
    audit_lineage(output,root,supervisor,bundle,checkpoint)
    return spec


def audit_lineage(output,root,supervisor,bundle,checkpoint):
    parent=verify_parent(root,supervisor,bundle,checkpoint);spec=read(output/'public/continuation.json')
    rows=read(output/'private/runs.json');plan=read(output/'public/plan.json')
    if (spec['schema']!='keyprint.balanced-continuation.v1' or spec['parent_files']!=parent['parent_files']
            or spec['checkpoint_sha256']!=parent['checkpoint_sha256'] or spec['supervisor_sha256']!=parent['supervisor_sha256']
            or spec['preserved_completed']!=len(parent['rows']) or spec['first_unstarted_index']!=len(parent['rows'])+1
            or spec['planned_new_attempts']!=127-len(parent['rows'])
            or spec['sealed_interruption_id']!=parent['interrupted']['review_id']
            or any(spec[k] is not False for k in ('quality_acceptance','detector_calibrated','launch_ready'))
            or plan!=dict(parent['plan'],continuation_sha256=digest(output/'public/continuation.json'))
            or len(rows)<spec['first_unstarted_index'] or len(rows)>128
            or rows[:len(parent['rows'])]!=parent['rows']):
        raise ValueError('Continuation lineage or preserved rows differ')
    if (read(output/'public/parent-checkpoint.json')!=read(checkpoint)
            or read(output/'public/parent-supervisor.json')!=read(supervisor/'result.json')):
        raise ValueError('Copied interruption receipts differ')
    for name,sha in spec['helpers_sha256'].items():
        if name not in HELPERS or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Continuation helper changed')
    if set(spec['helpers_sha256'])!=set(HELPERS):raise ValueError('Missing continuation helper binding')
    for name,sha in parent['parent_files'].items():
        if name not in ('public/plan.json','private/runs.json') and digest(output/name)!=sha:
            raise ValueError('Preserved parent artifact changed')
    row=parent['interrupted'];folder=output/'private'/row['review_id']
    if read(folder/'result.json')!=parent['sealed']:raise ValueError('Interruption cannot become a successful output')
    row['result_sha256']=digest(folder/'result.json')
    if rows[len(parent['rows'])]!=row:raise ValueError('Interrupted row changed or regenerated')
    if len({r['review_id'] for r in rows})!=len(rows):raise ValueError('Repeated review identity')
    for row,scheduled in zip(rows,plan['schedule'][:len(rows)],strict=True):
        if any(row[k]!=v for k,v in scheduled.items()):raise ValueError('Continuation schedule changed')
    return dict(continuation_sha256=digest(output/'public/continuation.json'),
                preserved_completed=len(parent['rows']),sealed_interrupted=1,new_attempts=len(rows)-spec['first_unstarted_index'],
                quality_acceptance=False,detector_calibrated=False,launch_ready=False)


def finalize(output,root,supervisor,bundle,checkpoint):
    if (output/'public/results.json').exists() or (output/'private/blind-review.json').exists():
        raise FileExistsError('Continuation already finalized or finalization interrupted; do not overwrite')
    lineage=audit_lineage(output,root,supervisor,bundle,checkpoint)
    rows=read(output/'private/runs.json')
    if len(rows)!=128:raise ValueError('All 128 scheduled outcomes required; no partial finalization')
    if read(output/'public/continuation.json')['new_model_load_authorized'] is not True:
        raise ValueError('Plan-only continuation cannot finalize execution')
    import random
    review=blind_packet(rows,read(output/'private/cases.json'),read(output/'private/rubrics.json'))
    random.SystemRandom().shuffle(review);save(output/'private/blind-review.json',review)
    result=dict(schema='keyprint.balanced-evaluation-results.v1',**totals(rows),
        plan_sha256=digest(output/'public/plan.json'),runs_sha256=digest(output/'private/runs.json'),
        identity_sha256=digest(output/'public/identity.json'),review_sha256=digest(output/'private/blind-review.json'),
        continuation=lineage,quality_acceptance=False,detector_calibrated=False,launch_ready=False)
    save(output/'public/results.json',result);checked=audit(output,bundle)
    save(output/'public/cohort-audit.json',dict(checked,continuation=lineage))
    return result
