"""Saved-path conditional likelihood diagnostic, not a shippable detector.

Requires privileged native base distributions and original prompts/paths. No
inference, token sampling, text rewriting, quality rerating or threshold search.
Ordinary paths get explicit marked counterfactual distributions; their stored
ordinary distributions must never masquerade as those counterfactuals.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics

from audit_paced_study import audit, digest
from paced_key_rank import write
from paced_source_session import PacedProfile, Config, partition, integer_distribution, IntegerDistribution


def metrics(base, marked, selected):
    """Exact integer normalization; binary64 arithmetic only for analysis."""
    for d in (base, marked):
        if (type(d.total) is not int or d.total <= 0 or not d.weights
                or any(type(w) is not int or w <= 0 for w in d.weights)
                or sum(d.weights) != d.total):
            raise ValueError('Positive normalized integer distributions required')
    if (len(base.weights) != len(marked.weights) or type(selected) is not int
            or not 0 <= selected < len(base.weights)):
        raise ValueError('Aligned support and selected index required')
    logs, p, q, tv = [], [], [], []
    for b, m in zip(base.weights, marked.weights):
        numerator, denominator = m * base.total, b * marked.total
        if 2*numerator < denominator or numerator > 2*denominator:
            raise ValueError('Candidate half-to-double bound violated')
        # Stable near identity; never feeds the sampling path.
        logs.append(math.log1p((numerator-denominator)/denominator))
        p.append(b/base.total); q.append(m/marked.total)
        tv.append(abs(numerator-denominator)/(base.total*marked.total))
    return {'selected_log_ratio': logs[selected],
            'kl_marked_to_base': math.fsum(v*l for v,l in zip(q,logs)),
            'kl_base_to_marked': -math.fsum(v*l for v,l in zip(p,logs)),
            'total_variation': .5*math.fsum(tv),
            'base_entropy_bits': -math.fsum(v*math.log2(v) for v in p if v),
            'marked_entropy_bits': -math.fsum(v*math.log2(v) for v in q if v)}


def same_distribution(a, b):
    return (len(a.weights) == len(b.weights)
            and all(x*b.total == y*a.total for x,y in zip(a.weights,b.weights)))


def inspect_path(path, row, profile, key):
    """Teacher-forced conditional evaluation, not fake sampled-session commits."""
    if digest(path) != row['journal_sha256']:
        raise ValueError('Recorded journal differs')
    context, used, tokens, values = (), set(), [], []
    pending = None
    with path.open() as f:
        for line in f:
            event = json.loads(line)
            if event['phase'] == 'prepared':
                if pending is not None or event['index'] != len(tokens):
                    raise ValueError('Out-of-order prepared step')
                decision = event['decision']
                if decision['mode'] != 'full_mark' or decision['protected_token_ids']:
                    raise ValueError('This diagnostic admits only the frozen general-purpose cohort')
                ids = event['token_ids']; raw = tuple(float.fromhex(h) for h in event['base_hex'])
                if (ids != sorted(set(ids)) or len(ids) != len(raw)
                        or any(not math.isfinite(v) or v <= 0 for v in raw)):
                    raise ValueError('Invalid sparse support')
                base = integer_distribution(raw)
                marked = (partition(raw,ids,profile,key,context,frozenset())[0]
                          if context not in used else base)
                saved = IntegerDistribution(tuple(event['distribution']['weights']), event['distribution']['total'])
                if not same_distribution(saved,marked if row['condition']=='marked' else base):
                    raise ValueError('Reconstructed distribution differs from generation')
                pending = (ids,base,marked)
            elif event['phase'] == 'committed':
                if pending is None or event['index'] != len(tokens):
                    raise ValueError('Out-of-order committed token')
                ids,base,marked = pending
                token = event['token_id']; index = ids.index(token)
                values.append(metrics(base,marked,index)); tokens.append(token)
                label = profile.label(token)
                if label is not None:
                    used.add(context)
                    context = (*context,label)[-profile.config.history:]
                pending = None
    if pending is not None or tokens != row['committed_token_ids'] or not tokens:
        raise ValueError('Incomplete or changed committed path')
    if row['completion'] != 'eos' or tokens[-1] not in profile.eos_ids:
        raise ValueError('Complete original EOS path required')
    sums = {k:math.fsum(v[k] for v in values) for k in values[0]}
    return {**{k:row[k] for k in ('review_id','case','condition','key_slot')},
            'steps':len(tokens),'sums':sums,
            'oracle_path_ratio_ge_100':sums['selected_log_ratio'] >= math.log(100),
            'distributions_reconciled':len(tokens)}


def summarize(rows):
    signatures={(r['case'],r['key_slot'],r['condition']) for r in rows}
    expected={(c,k,a) for c in {r['case'] for r in rows}
              for k in range(4) for a in ('ordinary','marked')}
    if (len(rows)!=128 or len({r['review_id'] for r in rows})!=128
            or len({r['case'] for r in rows})!=16 or signatures!=expected):
        raise ValueError('Complete unchanged 128-path cohort required')
    groups={}
    for arm in ('ordinary','marked'):
        group=[r for r in rows if r['condition']==arm]
        groups[arm]={'attempts':len(group),'steps':sum(r['steps'] for r in group),
            'mean_path_sums':{k:statistics.mean(r['sums'][k] for r in group) for k in group[0]['sums']},
            'median_path_log_ratio':statistics.median(r['sums']['selected_log_ratio'] for r in group),
            'oracle_path_ratio_ge_100':sum(r['oracle_path_ratio_ge_100'] for r in group)}
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('study','original','prior','model','output'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID'):
        raise RuntimeError('External watchdog required')
    checked=audit(args.study,args.original)
    if checked['audit_errors']: raise ValueError('Resolve cohort audit errors first')
    read=lambda f:json.loads(f.read_text())
    original=read(args.study/'public/plan.json');identity=read(args.study/'public/identity.json')
    rows=read(args.study/'private/runs.json')
    keys=[(args.prior/f'private/key-{i}').read_bytes() for i in range(4)]
    import hashlib
    if [hashlib.sha256(k).hexdigest() for k in keys]!=original['key_sha256']:
        raise ValueError('Generation keys changed')
    for name in ('tokenizer.json','tokenizer_config.json'):
        if digest(args.model/name)!=original['model_assets'][name]:
            raise ValueError('Tokenizer asset changed')
    for name in ('transformers','tokenizers','numpy'):
        if importlib.metadata.version(name)!=original['dependencies'][name]:
            raise ValueError('Tokenizer runtime changed')
    import keyprint
    package=Path(keyprint.__file__).parent
    sdk=lambda:{str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    if sdk()!=original['sdk_source_sha256']: raise ValueError('Frozen SDK changed')
    helper_names=('paced_information.py','paced_key_rank.py','audit_paced_study.py',
                  'paced_source_session.py','paced_integer_kernel.py')
    hashes=lambda:{n:digest(Path(__file__).with_name(n)) for n in helper_names}
    plan={'schema':'keyprint.paced-information-plan.v1','cohort':checked,'helpers_sha256':hashes(),
        'selection':'All 128 opened development paths; no omissions, new sampling or quality rerating',
        'oracle':'Sum log(marked/base) along each complete recorded path, including EOS; exact normalized integer ratios',
        'counterfactual':'Reconstruct marked distribution on ordinary prefixes; reconcile marked saved q and ordinary saved p',
        'cutoff':'Oracle path ratio >= 100; fixed descriptive diagnostic, not deployment calibration',
        'metrics':'Both conditional KL directions, total variation, base/marked entropy, selected log ratio',
        'scope':'Privileged saved native base weights and original paths. Shared kernel, no independent native-head replay. Reused English tasks/keys; conditional KL sums are realized compensators, not population KL bounds. Not a text detector or semantic-causation result.',
        'model_inference':False,'detector_calibrated':False,'quality_acceptance':False}
    args.output.mkdir(mode=0o700);write(args.output/'plan.json',plan)
    from transformers import AutoTokenizer
    from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding
    tok=AutoTokenizer.from_pretrained(str(args.model),trust_remote_code=False,local_files_only=True)
    binding=NFCWideByteLevelBinding.create(tok.backend_tokenizer.to_str(),vocabulary_size=248320,
        special_ids=tok.all_special_ids,eos_ids=[248046])
    profile=PacedProfile(binding.pieces,tokenizer_identity=binding.digest,eos_ids=binding.eos_ids,
                         config=Config(max_steps=1024))
    if binding.digest!=identity['binding_sha256'] or profile.digest!=identity['profile_sha256']:
        raise ValueError('Profile reconstruction differs')
    outcomes=[]
    with (args.output/'progress.jsonl').open('x') as f:
        for row in rows:
            value=inspect_path(args.study/'private'/row['review_id']/'journal.jsonl',row,profile,keys[row['key_slot']])
            outcomes.append(value);f.write(json.dumps(value)+'\n');f.flush()
            print(json.dumps({'completed':len(outcomes),'steps':sum(r['steps'] for r in outcomes)}),flush=True)
    if hashes()!=plan['helpers_sha256'] or sdk()!=original['sdk_source_sha256'] or audit(args.study,args.original)!=checked:
        raise ValueError('Frozen inputs or helper code changed')
    result={'schema':'keyprint.paced-information-results.v1','groups':summarize(outcomes),
        'rows':outcomes,'plan_sha256':digest(args.output/'plan.json'),
        'progress_sha256':digest(args.output/'progress.jsonl'),'scope':plan['scope'],
        'detector_calibrated':False,'quality_acceptance':False,'launch_ready':False}
    write(args.output/'results.json',result)
    print(json.dumps(result['groups']),flush=True)


if __name__=='__main__': main()
