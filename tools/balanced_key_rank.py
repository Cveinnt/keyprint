"""Frozen-decoy token-path diagnostic after complete source-only ratings freeze.

No model inference, threshold search, entropy selection or calibration claim.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from audit_paced_study import digest
from balanced_study import audit, read
from balanced_source_session import BalancedProfile, Config, replay_scores
from paced_key_rank import compile_paths, score_paths, key_rank, summarize, write
from summarize_balanced_quality import summarize_frozen


def evaluate(profile, rows, keys, decoys, emit=lambda item: None):
    if len(keys)!=4 or len(decoys)!=199 or len(set(keys+decoys))!=203:
        raise ValueError('Exactly four distinct owner keys and 199 disjoint frozen decoys required')
    if any(type(k) is not bytes or len(k)!=32 for k in keys+decoys):
        raise ValueError('Exactly 32 bytes per key required')
    eligible=[r['completion']=='eos' and not any(k in r for k in
              ('error_type','decode_error','audit_error')) for r in rows]
    paths=[r['committed_token_ids'] if ok else [] for r,ok in zip(rows,eligible,strict=True)]
    messages,indexes=compile_paths(profile,paths)
    owners=[score_paths(k,messages,indexes) for k in keys]
    for i,(row,ok) in enumerate(zip(rows,eligible,strict=True)):
        if not ok:continue
        for offset,saved in enumerate(row['raw_counts']):
            slot=(row['key_slot']+offset)%4
            replay=replay_scores(profile,keys[slot],paths[i])
            if (replay!=saved or owners[slot][i]!=saved['ones']
                    or len(indexes[i])*30!=saved['trials']):
                raise ValueError('Compiled score does not match retained and independent replay')
    null=[]
    for i,key in enumerate(decoys):
        values=score_paths(key,messages,indexes);null.append(values)
        emit(dict(index=i,ones=values))
    result=[]
    for i,(row,ok) in enumerate(zip(rows,eligible,strict=True)):
        item={k:row[k] for k in ('review_id','case','condition','key_slot')}
        trials=len(indexes[i])*30
        if not ok or not trials:
            item['unavailable']='Failed, capped, undecodable or no eligible events; not a clean negative'
        else:
            for field,slot in (('rank',row['key_slot']),('next_key_rank',(row['key_slot']+1)%4)):
                item[field]=key_rank(owners[slot][i],[n[i] for n in null],trials)
            item.update(ones=owners[row['key_slot']][i],next_key_ones=owners[(row['key_slot']+1)%4][i],trials=trials)
        result.append(item)
    return result


def aggregate(rows,cases):
    result=summarize(rows)
    def next_view(items):
        return [dict({k:v for k,v in r.items() if k not in ('rank','next_key_rank')},
                     **({'rank':r['next_key_rank']} if 'next_key_rank' in r else {})) for r in items]
    result['next_key_groups']=summarize(next_view(rows))['groups']
    result['by_language']={}
    for lang in sorted({c['language'] for c in cases}):
        ids={c['id'] for c in cases if c['language']==lang}
        selected=[r for r in rows if r['case'] in ids]
        result['by_language'][lang]={arm:{
            'attempts':sum(r['condition']==arm for r in selected),
            'available':sum(r['condition']==arm and 'rank' in r for r in selected),
            'matching_hits':sum(r['condition']==arm and r.get('rank',{}).get('at_or_below_one_percent',False) for r in selected),
            'next_key_hits':sum(r['condition']==arm and r.get('next_key_rank',{}).get('at_or_below_one_percent',False) for r in selected)
        } for arm in ('ordinary','marked')}
    result['rows']=rows
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','bundle','frozen','model','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if str(os.getppid())!=os.environ.get('KEYPRINT_WATCHDOG_PID'):
        raise RuntimeError('External memory watchdog required for offline scoring')
    checked=audit(args.study,args.bundle)
    # Enforce the declared review order before exposing any decoy ranks.
    summarize_frozen(args.study,args.bundle,args.frozen)
    plan=read(args.study/'public/plan.json');identity=read(args.study/'public/identity.json')
    assets={n:digest(args.model/n) for n in ('tokenizer.json','tokenizer_config.json')}
    if any(h!=plan['model_assets'][n] for n,h in assets.items()):raise ValueError('Tokenizer assets changed')
    for name in ('transformers','tokenizers','numpy'):
        if importlib.metadata.version(name)!=plan['dependencies'][name]:raise ValueError('Pinned tokenizer runtime differs')
    import keyprint
    package=Path(keyprint.__file__).parent
    if {str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}!=plan['sdk_source_sha256']:
        raise ValueError('Frozen SDK changed')
    from transformers import AutoTokenizer
    from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding
    tokenizer=AutoTokenizer.from_pretrained(str(args.model),trust_remote_code=False,local_files_only=True)
    binding=NFCWideByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),vocabulary_size=248320,
                                          special_ids=tokenizer.all_special_ids,eos_ids=[248046])
    profile=BalancedProfile(binding.pieces,tokenizer_identity=binding.digest,eos_ids=binding.eos_ids,config=Config(max_steps=1024))
    if profile.digest!=identity['profile_sha256'] or binding.digest!=identity['binding_sha256']:
        raise ValueError('Reconstructed profile differs')
    keys=[(args.bundle/f'private/key-{i}').read_bytes() for i in range(4)]
    decoys=[bytes.fromhex(k) for k in read(args.bundle/'private/decoys.json')]
    helpers={n:digest(Path(__file__).with_name(n)) for n in
             ('balanced_key_rank.py','paced_key_rank.py','balanced_source_session.py','balanced_study.py')}
    args.output.mkdir(mode=0o700);(args.output/'private').mkdir(mode=0o700);(args.output/'public').mkdir()
    receipt=dict(schema='keyprint.balanced-key-rank-plan.v1',cohort=checked,
        helpers_sha256=helpers,ratings_commitment_sha256=digest(args.frozen/'commitment.json'),
        protocol_sha256=digest(args.bundle/'public/protocol.json'),
        model_loaded=False,detector_calibrated=False,quality_acceptance=False,launch_ready=False)
    write(args.output/'public/plan.json',receipt)
    with (args.output/'private/decoy-scores.jsonl').open('x') as f:
        def emit(item):
            f.write(json.dumps(item)+'\n');f.flush()
            if (item['index']+1)%10==0:print(json.dumps({'completed_decoys':item['index']+1}),flush=True)
        rows=evaluate(profile,read(args.study/'private/runs.json'),keys,decoys,emit)
    if (audit(args.study,args.bundle)!=checked
            or any(digest(Path(__file__).with_name(n))!=h for n,h in helpers.items())):
        raise ValueError('Inputs or code changed during diagnostic')
    summarize_frozen(args.study,args.bundle,args.frozen)
    result=aggregate(rows,read(args.bundle/'private/cases.json'))
    result.update(plan_sha256=digest(args.output/'public/plan.json'),
                  decoy_scores_sha256=digest(args.output/'private/decoy-scores.jsonl'),
                  scope=read(args.bundle/'public/protocol.json')['detection'])
    write(args.output/'public/results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
