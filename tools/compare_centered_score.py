"""Paired saved-prefix comparison; no new text or production detector claims.

All opened marked paths, same old keyed labels and native base probabilities.
New rules are counterfactual conditional distributions, not generated paths or
fresh-profile performance. Ordinary paths are not needed for this comparison.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from audit_paced_study import audit, digest
from centered_score import tilt, expected_score_lift, POLICY
from paced_information import metrics
from paced_regimes import entropy_bin, LABELS
from paced_key_rank import write
from paced_source_session import PacedProfile, Config, integer_distribution, IntegerDistribution


def cell():
    return dict(steps=0, old_kl=0., new_kl=0., old_tv=0., new_tv=0.,
                old_score_lift=0., new_score_lift=0.,
                new_kl_greater_steps=0, new_tv_greater_steps=0,
                new_score_lift_greater_steps=0)


def inspect(path, row, profile, key, rule=tilt):
    if digest(path)!=row['journal_sha256']: raise ValueError('Journal changed')
    used,context,tokens,pending=set(),(),[],None
    bins={name:cell() for name in LABELS}
    with path.open() as f:
        for line in f:
            e=json.loads(line)
            if e['phase']=='prepared':
                if pending is not None or e['index']!=len(tokens): raise ValueError('Prepared order differs')
                decision=e['decision']
                if decision['mode']!='full_mark' or decision['protected_token_ids']:
                    raise ValueError('Only original general-purpose cohort admitted')
                ids=e['token_ids'];raw=tuple(float.fromhex(h) for h in e['base_hex'])
                if ids!=sorted(set(ids)) or len(ids)!=len(raw) or any(not math.isfinite(v) or v<=0 for v in raw):
                    raise ValueError('Invalid sparse support')
                base=integer_distribution(raw)
                old=IntegerDistribution(tuple(e['distribution']['weights']),e['distribution']['total'])
                labels={profile.classes[i] for i in ids if profile.classes[i] is not None}
                scores_by_label=({label:sum(profile.bits(key,context,label)) for label in labels}
                                 if context not in used else {})
                scores=[scores_by_label.get(profile.classes[i]) for i in ids]
                new=rule(base,scores,profile.config.layers)
                old_metrics=metrics(base,old,0);new_metrics=metrics(base,new,0)
                values=dict(old_kl=old_metrics['kl_marked_to_base'],new_kl=new_metrics['kl_marked_to_base'],
                            old_tv=old_metrics['total_variation'],new_tv=new_metrics['total_variation'],
                            old_score_lift=expected_score_lift(base,old,scores,profile.config.layers),
                            new_score_lift=expected_score_lift(base,new,scores,profile.config.layers))
                b=bins[entropy_bin(old_metrics['base_entropy_bits'])]
                b['steps']+=1
                for k,v in values.items(): b[k]+=v
                for kind in ('kl','tv','score_lift'):
                    b[f'new_{kind}_greater_steps']+=values[f'new_{kind}']>values[f'old_{kind}']
                pending=ids
            elif e['phase']=='committed':
                if pending is None or e['index']!=len(tokens) or e['token_id'] not in pending:
                    raise ValueError('Committed order differs')
                token=e['token_id'];tokens.append(token);pending=None
                label=profile.label(token)
                if label is not None:
                    used.add(context);context=(*context,label)[-profile.config.history:]
    if pending is not None or tokens!=row['committed_token_ids'] or row['completion']!='eos' or tokens[-1] not in profile.eos_ids:
        raise ValueError('Incomplete original EOS path')
    return {'review_id':row['review_id'],'steps':len(tokens),'bins':bins,
            'sums':{k:math.fsum(b[k] for b in bins.values()) for k in cell()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('study','original','information','prior','model','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--rule',choices=('linear','balanced'),default='linear')
    args=p.parse_args()
    from balanced_score import allocate, POLICY as BALANCED_POLICY
    rule=tilt if args.rule=='linear' else allocate
    policy=POLICY if args.rule=='linear' else BALANCED_POLICY
    if str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID'):raise RuntimeError('External watchdog required')
    checked=audit(args.study,args.original)
    if checked['audit_errors'] or checked['eos']!=128:raise ValueError('Complete audited cohort required')
    read=lambda p:json.loads(p.read_text())
    original=read(args.study/'public/plan.json');identity=read(args.study/'public/identity.json')
    old_plan=read(args.information/'plan.json');old_result=read(args.information/'results.json')
    if old_plan['cohort']!=checked or old_result['plan_sha256']!=digest(args.information/'plan.json'):
        raise ValueError('Prior information result changed')
    keys=[(args.prior/f'private/key-{i}').read_bytes() for i in range(4)]
    if [hashlib.sha256(k).hexdigest() for k in keys]!=original['key_sha256']:raise ValueError('Keys changed')
    for n in ('tokenizer.json','tokenizer_config.json'):
        if digest(args.model/n)!=original['model_assets'][n]:raise ValueError('Tokenizer changed')
    for n in ('transformers','tokenizers','numpy'):
        if importlib.metadata.version(n)!=original['dependencies'][n]:raise ValueError('Tokenizer runtime changed')
    import keyprint
    package=Path(keyprint.__file__).parent
    sdk=lambda:{str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    if sdk()!=original['sdk_source_sha256']:raise ValueError('SDK source changed')
    names=('compare_centered_score.py','centered_score.py','balanced_score.py','paced_information.py','paced_regimes.py',
           'paced_source_session.py','paced_integer_kernel.py','paced_key_rank.py','audit_paced_study.py')
    bindings=lambda:{n:digest(Path(__file__).with_name(n)) for n in names}
    plan=dict(schema='keyprint.centered-score-comparison-plan.v1',policy=policy,
              cohort=checked,helpers_sha256=bindings(),prior_information_sha256=digest(args.information/'results.json'),
              selection='All 64 opened marked paths; every saved prefix including EOS and repeated contexts',
              numeric='Exact normalized integers; global ratio [1/2,3/2]; excluded probabilities unchanged',
              labels='Same original 30-layer keyed labels for paired comparison only, not a new generation profile',
              replaced_constraint='No per-layer p/8 pacing; the new rule is a single aggregate-score adjustment',
              scoring='Uniform mean of keyed layer bits; no native entropy used for selection or scoring',
              scope=__doc__,model_inference=False,detector_calibrated=False,quality_acceptance=False)
    args.output.mkdir(mode=0o700);write(args.output/'plan.json',plan)
    from transformers import AutoTokenizer
    from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding
    tok=AutoTokenizer.from_pretrained(str(args.model),trust_remote_code=False,local_files_only=True)
    binding=NFCWideByteLevelBinding.create(tok.backend_tokenizer.to_str(),vocabulary_size=248320,
                                         special_ids=tok.all_special_ids,eos_ids=[248046])
    profile=PacedProfile(binding.pieces,tokenizer_identity=binding.digest,eos_ids=binding.eos_ids,config=Config(max_steps=1024))
    if binding.digest!=identity['binding_sha256'] or profile.digest!=identity['profile_sha256']:
        raise ValueError('Old paired-label profile differs')
    rows=[r for r in read(args.study/'private/runs.json') if r['condition']=='marked']
    prior={r['review_id']:r for r in old_result['rows'] if r['condition']=='marked'}
    if len(rows)!=64 or set(prior)!={r['review_id'] for r in rows}:raise ValueError('Incomplete marked cohort')
    output=[]
    with (args.output/'progress.jsonl').open('x') as f:
        for row in rows:
            value=inspect(args.study/'private'/row['review_id']/'journal.jsonl',row,profile,keys[row['key_slot']],rule)
            for new,old in [('old_kl','kl_marked_to_base'),('old_tv','total_variation')]:
                if not math.isclose(value['sums'][new],prior[row['review_id']]['sums'][old],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Old conditional metric mismatch')
            output.append(value);f.write(json.dumps(value)+'\n');f.flush()
            print(json.dumps({'completed':len(output)}),flush=True)
    if bindings()!=plan['helpers_sha256'] or sdk()!=original['sdk_source_sha256'] or audit(args.study,args.original)!=checked:
        raise ValueError('Frozen inputs changed')
    bins={name:{k:math.fsum(r['bins'][name][k] for r in output) for k in cell()} for name in LABELS}
    total={k:math.fsum(b[k] for b in bins.values()) for k in cell()}
    result=dict(schema='keyprint.centered-score-comparison-results.v1',rows=output,bins=bins,total=total,
                plan_sha256=digest(args.output/'plan.json'),progress_sha256=digest(args.output/'progress.jsonl'),
                launch_ready=False,detector_calibrated=False,quality_acceptance=False,scope=__doc__)
    write(args.output/'results.json',result);print(json.dumps(total),flush=True)


if __name__=='__main__':main()
