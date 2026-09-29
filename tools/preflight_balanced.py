"""Four fresh native prefix attempts under the balanced research profile.

Execution preflight only: short capped prefixes cannot establish task quality,
language preservation, detection power or serving compatibility.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import uuid

from audit_paced_study import digest
from balanced_inference import generate, audit_output, save
from balanced_source_session import BalancedProfile, Config, policy_spec, replay_scores

CASES=(
    dict(id='prefix-en',language='English',prompt=(
        'Write an English email of about 80 words using only these facts. '
        'The robotics workshop is on October 14 at 10:30 in Room 6. '
        'Registration closes on October 9. Equipment is supplied. '
        'Attendance is free, but registration is required. Do not invent any details.')),
    dict(id='prefix-es',language='Spanish',prompt=(
        'Redacta un correo en español de unas 80 palabras usando solo estos datos. '
        'La visita al observatorio es el 18 de noviembre a las 19:00. '
        'La entrada cuesta 12 euros. Los menores de 16 años deben ir con un adulto. '
        'La visita se cancela si llueve. No inventes información.')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','original','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID') or 'KEYPRINT_MLX_MEMORY_LOG' not in os.environ:
        raise RuntimeError('External watchdog and cache-disabled MLX wrapper required')
    read=lambda p:json.loads(p.read_text())
    reference=read(args.original/'public/identity.json');old_plan=read(args.original/'public/plan.json')
    import keyprint
    from keyprint.experimental.wide_mlx import verify_assets, NFCWideByteLevelBinding
    package=Path(keyprint.__file__).parent
    sdk=lambda:{str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    if sdk()!=old_plan['sdk_source_sha256'] or verify_assets(args.model)!=reference['model_assets']:
        raise ValueError('Pinned SDK or model assets changed')
    for name,version in reference['dependencies'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Pinned runtime changed')
    helpers=('preflight_balanced.py','balanced_inference.py','balanced_source_session.py',
             'balanced_score.py','centered_score.py','paced_source_session.py',
             'paced_integer_kernel.py','replay_wide_mlx.py','memory_watchdog.py','mlx_guarded_worker.py')
    hashes=lambda:{n:digest(Path(__file__).with_name(n)) for n in helpers}
    args.output.mkdir(mode=0o700);public=args.output/'public';private=args.output/'private'
    public.mkdir();private.mkdir(mode=0o700)
    keys=[os.urandom(32) for _ in CASES]
    for i,key in enumerate(keys):(private/f'key-{i}').write_bytes(key)
    save(private/'cases.json',CASES)
    schedule=[dict(case=c['id'],key_slot=i,condition=arm) for i,c in enumerate(CASES)
              for arm in ('ordinary','marked')]
    plan=dict(schema='keyprint.balanced-native-preflight-plan.v1',stage='preflight',
              scripts_sha256=hashes(),sdk_source_sha256=sdk(),model_assets=reference['model_assets'],
              dependencies=reference['dependencies'],policy=policy_spec(),schedule=schedule,
              cases_sha256=digest(private/'cases.json'),
              key_sha256=[hashlib.sha256(k).hexdigest() for k in keys],
              settings=dict(max_tokens=16,temperature=.7,top_k=100,profile_max_steps=1024,enable_thinking=False),
              random_bits='Fresh OS SystemRandom; retained draw transcripts; no paired shared RNG',
              failure_policy='One attempt each; no retry, text repair or output selection; all caps/errors retained',
              scope=__doc__,quality_acceptance=False,detector_calibrated=False)
    save(public/'plan.json',plan)
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
                            config=Config(max_steps=1024))
    save(public/'identity.json',dict(profile_sha256=profile.digest,binding_sha256=binding.digest))
    prompts={c['id']:tokenizer.apply_chat_template([{'role':'user','content':c['prompt']}],
             tokenize=True,add_generation_prompt=True,enable_thinking=False) for c in CASES}
    save(private/'prompt-ids.json',prompts)
    rng=random.SystemRandom();outcomes=[]
    for attempt in schedule:
        row=dict(attempt,review_id=uuid.uuid4().hex[:12]);cache=make_prompt_cache(model)
        def forward(ids):
            logits=model(mx.array([ids]),cache=cache)[:,-1,:].astype(mx.float32)
            mx.eval(logits);return np.array(logits)
        try:
            result=generate(profile=profile,binding=binding,key=keys[row['key_slot']],condition=row['condition'],
                prompt_ids=prompts[row['case']],forward=forward,
                decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                random_bits=rng.getrandbits,output=private/row['review_id'],max_tokens=16)
            row.update(result);row['result_sha256']=digest(private/row['review_id']/'result.json')
            if 'error_type' not in row and 'decode_error' not in row:
                try:row['audit']=audit_output(private/row['review_id'],profile,keys[row['key_slot']],binding=binding)
                except Exception as error:row['audit_error']=dict(type=type(error).__name__,message=str(error))
            row['raw_score']=replay_scores(profile,keys[row['key_slot']],row['committed_token_ids'])
        finally:
            cache.clear();mx.synchronize();mx.clear_cache()
        outcomes.append(row);save(private/'runs.json',outcomes)
        print(json.dumps({'completed':len(outcomes),'completion':row['completion'],
                          'tokens':len(row['committed_token_ids']),'error':row.get('error_type')}),flush=True)
    if hashes()!=plan['scripts_sha256'] or sdk()!=plan['sdk_source_sha256']:
        raise ValueError('Code changed during preflight')
    failures=sum(any(k in r for k in ('error_type','decode_error','audit_error')) for r in outcomes)
    result=dict(schema='keyprint.balanced-native-preflight-results.v1',attempts=len(outcomes),
                failures=failures,audited_runs=sum('audit' in r for r in outcomes),
                committed_tokens=sum(len(r['committed_token_ids']) for r in outcomes),
                caps=sum(r['completion']=='cap' for r in outcomes),eos=sum(r['completion']=='eos' for r in outcomes),
                plan_sha256=digest(public/'plan.json'),runs_sha256=digest(private/'runs.json'),
                identity_sha256=digest(public/'identity.json'),quality_acceptance=False,
                detector_calibrated=False,launch_ready=False,scope=__doc__)
    save(public/'results.json',result);print(json.dumps(result),flush=True)
    if failures:raise RuntimeError('Preflight failures retained; do not promote')


if __name__=='__main__':main()
