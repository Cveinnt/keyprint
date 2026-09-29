"""Execute the frozen multilingual study without retrying or selecting outputs.

Plan-only never imports MLX or loads a model. Execution requires a completed
balanced native preflight and the external memory/cache guards.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import uuid

from audit_paced_study import digest
from balanced_inference import generate,audit_output,save
from balanced_source_session import BalancedProfile,Config,replay_scores
from balanced_study import read,validate_preflight,blind_packet,totals,audit
from freeze_balanced_evaluation import validate

HELPERS=('run_balanced_evaluation.py','balanced_study.py','freeze_balanced_evaluation.py',
         'balanced_inference.py','balanced_source_session.py','balanced_score.py','centered_score.py',
         'paced_source_session.py','paced_integer_kernel.py','replay_wide_mlx.py',
         'preflight_balanced.py','validate_source_grounded.py','memory_watchdog.py','mlx_guarded_worker.py')


def make_plan(bundle,reference,original,*,execute,preflight=None):
    manifest=validate(bundle);protocol=read(bundle/'public/protocol.json')
    if execute and preflight is None:raise ValueError('Completed preflight required for execution')
    if preflight is not None and any(preflight[k]!=reference[k] for k in ('model_assets','dependencies')):
        raise ValueError('Preflight model or runtime differs')
    if preflight is not None and preflight['sdk_source_sha256']!=original['sdk_source_sha256']:
        raise ValueError('Preflight SDK differs')
    return dict(schema='keyprint.balanced-evaluation-run.v1',stage='study',
                model_load_authorized_in_this_invocation=execute,
                bundle_manifest_sha256=digest(bundle/'public/manifest.json'),
                bundle_protocol_sha256=digest(bundle/'public/protocol.json'),
                schedule=manifest['schedule'],policy=manifest['policy'],settings=protocol['settings'],
                scripts_sha256={n:digest(Path(__file__).with_name(n)) for n in HELPERS},
                sdk_source_sha256=original['sdk_source_sha256'],model_assets=reference['model_assets'],
                dependencies=reference['dependencies'],preflight=preflight,
                failure_policy=protocol['failure_policy'],scope=protocol['scope'],
                random_bits='Fresh OS SystemRandom per draw; pairs do not share randomness',
                quality_acceptance=False,detector_calibrated=False,launch_ready=False)


def score_attempt(row,profile,keys):
    if any(k in row for k in ('error_type','decode_error','audit_error')) or row['completion']!='eos':
        return [dict(unavailable='Failed, capped or undecodable attempt; not a negative control') for _ in range(2)]
    return [replay_scores(profile,keys[slot],row['committed_token_ids'])
            for slot in (row['key_slot'],(row['key_slot']+1)%4)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('bundle','model','original','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--preflight',type=Path);p.add_argument('--preflight-supervisor',type=Path)
    args=p.parse_args()
    checked=validate(args.bundle)
    reference=read(args.original/'public/identity.json');original=read(args.original/'public/plan.json')
    before=None
    if not args.plan_only:
        if str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID') or 'KEYPRINT_MLX_MEMORY_LOG' not in os.environ:
            raise RuntimeError('External watchdog and cache-disabled MLX wrapper required')
        if args.preflight is None or args.preflight_supervisor is None:
            raise ValueError('Completed balanced preflight and supervisor receipt required')
        before=validate_preflight(args.preflight,args.preflight_supervisor)
    plan=make_plan(args.bundle,reference,original,execute=not args.plan_only,preflight=before)
    import keyprint
    package=Path(keyprint.__file__).parent
    sdk=lambda:{str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    if sdk()!=plan['sdk_source_sha256']:raise ValueError('Pinned SDK changed')
    for n,v in plan['dependencies'].items():
        if importlib.metadata.version(n)!=v:raise ValueError('Pinned runtime differs')
    args.output.mkdir(mode=0o700);public=args.output/'public';private=args.output/'private'
    public.mkdir();private.mkdir(mode=0o700)
    save(public/'plan.json',plan)
    cases=read(args.bundle/'private/cases.json');rubrics=read(args.bundle/'private/rubrics.json')
    save(private/'cases.json',cases);save(private/'rubrics.json',rubrics)
    if args.plan_only:
        print(json.dumps({'planned_attempts':128,'model_loaded':False,'generation_executed':False}));return
    from keyprint.experimental.wide_mlx import verify_assets,NFCWideByteLevelBinding
    if verify_assets(args.model)!=plan['model_assets']:raise ValueError('Pinned model differs')
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    if mx.set_cache_limit(0)!=0:raise RuntimeError('MLX cache must already be disabled')
    model,tokenizer=load(str(args.model),tokenizer_config={'trust_remote_code':False,'local_files_only':True})
    if set(tokenizer.eos_token_ids)!={248046}:raise ValueError('Runtime EOS differs')
    binding=NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),vocabulary_size=248320,
                                         special_ids=tokenizer.all_special_ids,eos_ids=[248046])
    profile=BalancedProfile(binding.pieces,tokenizer_identity=binding.digest,eos_ids=binding.eos_ids,
                            config=Config(max_steps=plan['settings']['profile_max_steps']))
    if profile.digest!=before['profile_sha256']:raise ValueError('Loaded profile differs from preflight')
    save(public/'identity.json',dict(profile_sha256=profile.digest,binding_sha256=binding.digest))
    prompts={c['id']:tokenizer.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=True,
              add_generation_prompt=True,enable_thinking=False) for c in cases}
    save(private/'prompt-ids.json',prompts)
    keys=[(args.bundle/f'private/key-{i}').read_bytes() for i in range(4)]
    outcomes=[];rng=random.SystemRandom()
    for attempt in plan['schedule']:
        row=dict(attempt,review_id=uuid.uuid4().hex[:12]);cache=make_prompt_cache(model)
        def forward(ids):
            logits=model(mx.array([ids]),cache=cache)[:,-1,:].astype(mx.float32)
            mx.eval(logits);return np.array(logits)
        try:
            result=generate(profile=profile,binding=binding,key=keys[row['key_slot']],condition=row['condition'],
                prompt_ids=prompts[row['case']],forward=forward,
                decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                random_bits=rng.getrandbits,output=private/row['review_id'],max_tokens=plan['settings']['max_tokens'],
                temperature=plan['settings']['temperature'],top_k=plan['settings']['top_k'])
            row.update(result);row['result_sha256']=digest(private/row['review_id']/'result.json')
            if 'error_type' not in row and 'decode_error' not in row:
                try:row['audit']=audit_output(private/row['review_id'],profile,keys[row['key_slot']],binding=binding)
                except Exception as error:row['audit_error']=dict(type=type(error).__name__,message=str(error))
            row['raw_counts']=score_attempt(row,profile,keys)
        finally:
            cache.clear();mx.synchronize();mx.clear_cache()
        outcomes.append(row);save(private/'runs.json',outcomes)
        print(json.dumps({'completed':len(outcomes),'ending':row['completion'],
                          'error':row.get('error_type'),'tokens':len(row['committed_token_ids'])}),flush=True)
    if (validate(args.bundle)!=checked or sdk()!=plan['sdk_source_sha256']
            or any(digest(Path(__file__).with_name(n))!=h for n,h in plan['scripts_sha256'].items())):
        raise ValueError('Frozen inputs or code changed during generation')
    review=blind_packet(outcomes,cases,rubrics);rng.shuffle(review);save(private/'blind-review.json',review)
    result=dict(schema='keyprint.balanced-evaluation-results.v1',**totals(outcomes),
                plan_sha256=digest(public/'plan.json'),runs_sha256=digest(private/'runs.json'),
                identity_sha256=digest(public/'identity.json'),review_sha256=digest(private/'blind-review.json'),
                quality_acceptance=False,detector_calibrated=False,launch_ready=False)
    save(public/'results.json',result)
    save(public/'cohort-audit.json',audit(args.output,args.bundle))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
