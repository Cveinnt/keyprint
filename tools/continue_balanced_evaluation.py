"""One guarded continuation of only the frozen study's unstarted attempts.

The interrupted attempt remains a failure. Never regenerate completed text,
resume a failed continuation implicitly, or change sampling/quality criteria.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import uuid

from audit_paced_study import digest
from balanced_study import read,validate_preflight
from balanced_continuation import prepare,audit_lineage,finalize
from balanced_inference import generate,audit_output,save
from balanced_source_session import BalancedProfile,Config
from run_balanced_evaluation import score_attempt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','parent-supervisor','bundle','checkpoint','model','preflight','preflight-supervisor','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    if not args.plan_only and (str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID')
                              or 'KEYPRINT_MLX_MEMORY_LOG' not in os.environ):
        raise RuntimeError('External watchdog and cache-disabled MLX wrapper required')
    preflight=validate_preflight(args.preflight,args.preflight_supervisor)
    parent_plan=read(args.parent/'public/plan.json')
    if preflight!=parent_plan['preflight']:raise ValueError('Original successful preflight changed')
    import keyprint
    package=Path(keyprint.__file__).parent
    sdk=lambda:{str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    if sdk()!=parent_plan['sdk_source_sha256']:raise ValueError('Frozen SDK changed')
    for n,v in parent_plan['dependencies'].items():
        if importlib.metadata.version(n)!=v:raise ValueError('Frozen runtime changed')
    spec=prepare(args.parent,args.parent_supervisor,args.bundle,args.checkpoint,args.output,execute=not args.plan_only)
    if args.plan_only:
        print(json.dumps({'preserved':spec['preserved_completed'],'sealed_interrupted':1,
                          'planned_new':spec['planned_new_attempts'],'model_loaded':False}));return
    from keyprint.experimental.wide_mlx import verify_assets,NFCWideByteLevelBinding
    if verify_assets(args.model)!=parent_plan['model_assets']:raise ValueError('Frozen model changed')
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
                            config=Config(max_steps=parent_plan['settings']['profile_max_steps']))
    identity=read(args.output/'public/identity.json')
    if identity!=dict(profile_sha256=profile.digest,binding_sha256=binding.digest):raise ValueError('Loaded profile differs')
    private=args.output/'private';cases=read(private/'cases.json');prompts=read(private/'prompt-ids.json')
    actual={c['id']:tokenizer.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=True,
            add_generation_prompt=True,enable_thinking=False) for c in cases}
    if actual!=prompts:raise ValueError('Original prompt tokenization changed')
    keys=[(args.bundle/f'private/key-{i}').read_bytes() for i in range(4)]
    rows=read(private/'runs.json');settings=parent_plan['settings'];rng=random.SystemRandom()
    for attempt in parent_plan['schedule'][spec['first_unstarted_index']:]:
        row=dict(attempt,review_id=uuid.uuid4().hex[:12]);cache=make_prompt_cache(model)
        def forward(ids):
            logits=model(mx.array([ids]),cache=cache)[:,-1,:].astype(mx.float32)
            mx.eval(logits);return np.array(logits)
        try:
            result=generate(profile=profile,binding=binding,key=keys[row['key_slot']],condition=row['condition'],
                prompt_ids=prompts[row['case']],forward=forward,
                decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                random_bits=rng.getrandbits,output=private/row['review_id'],max_tokens=settings['max_tokens'],
                temperature=settings['temperature'],top_k=settings['top_k'])
            row.update(result);row['result_sha256']=digest(private/row['review_id']/'result.json')
            if 'error_type' not in row and 'decode_error' not in row:
                try:row['audit']=audit_output(private/row['review_id'],profile,keys[row['key_slot']],binding=binding)
                except Exception as error:row['audit_error']=dict(type=type(error).__name__,message=str(error))
            row['raw_counts']=score_attempt(row,profile,keys)
        finally:
            cache.clear();mx.synchronize();mx.clear_cache()
        rows.append(row);save(private/'runs.json',rows)
        print(json.dumps({'recorded_outcomes':len(rows),'new_attempts':len(rows)-spec['first_unstarted_index'],
                          'ending':row['completion'],'error':row.get('error_type'),'tokens':len(row['committed_token_ids'])}),flush=True)
    if sdk()!=parent_plan['sdk_source_sha256']:raise ValueError('SDK changed during continuation')
    checked=finalize(args.output,args.parent,args.parent_supervisor,args.bundle,args.checkpoint)
    print(json.dumps({k:v for k,v in checked.items() if k not in ('plan_sha256','runs_sha256','identity_sha256','review_sha256')}),flush=True)


if __name__=='__main__':main()
