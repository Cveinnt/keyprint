"""Check a cache-disabled MLX policy against three already measured null cases.

Select smallest, median and largest retained conditioning-plus-text token counts.
Require byte-identical raw-head hashes, complete sparse probabilities and scores.
No previously unmeasured source task enters this resource-policy preflight.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import time

from prompt_confirmation_selection import sha
from prompt_null_selection import prompt
from prompt_conditioned_likelihood import measure
from surrogate_likelihood import score
from develop_surrogate_likelihood import write

FLOOR = 2 * 1024**3


def select_completed(study, rows):
    choices = []
    for row in rows:
        if 'error' in row: continue
        path = study / (row['id'] + '.heads.json')
        if sha(path.read_bytes()) != row['heads_sha256']:
            raise ValueError('Original heads differ')
        data = json.loads(path.read_text())
        choices.append((len(data['conditioning_prefix_ids']) + len(data['token_ids']), row['id'], row))
    if len(choices) < 3: raise ValueError('Three measured controls required')
    choices.sort(key=lambda item: item[:2])
    return [choices[index][2] for index in (0, len(choices)//2, len(choices)-1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','model','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); public = args.study/'public'
    plan = json.loads((public/'plan.json').read_text()); audit = json.loads((public/'integrity.json').read_text())
    if audit['status'] != 'pass': raise ValueError('Audited parent study required')
    for name, field in (('plan.json','plan_sha256'),('summary.json','summary_sha256')):
        if sha((public/name).read_bytes()) != audit[field]: raise ValueError('Audited parent file differs')
    raw = (public/'results.jsonl').read_bytes()
    if len(raw) != audit['results_prefix_bytes'] or sha(raw) != audit['results_prefix_sha256']:
        raise ValueError('Complete audited parent results required')
    if sha(args.source.read_bytes()) != plan['source_sha256']: raise ValueError('Source differs')
    for name, digest in plan['dependencies_sha256'].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest: raise ValueError('Frozen dependency differs')
    selected = select_completed(args.study, [json.loads(line) for line in raw.splitlines()])
    source = [json.loads(line) for line in args.source.read_bytes().splitlines()]
    args.output.mkdir(mode=0o700); result_dir = args.output/'public'; result_dir.mkdir()
    declaration = {'scope':__doc__, 'script_sha256':sha(Path(__file__).read_bytes()),
        'parent':str(args.study.resolve()),'parent_audit_sha256':sha((public/'integrity.json').read_bytes()),
        'dependencies_sha256':plan['dependencies_sha256'], 'cache_limit_bytes':0,'clear_between_documents':True,
        'disk_floor_bytes':FLOOR,'cases':[r['id'] for r in selected],
        'criterion':'All raw heads, sparse probabilities, literal tokens and both full score records match exactly; free space remains >=2 GiB',
        'limits':'Three existing tasks only; not a timing benchmark or proof of parity on every workload'}
    write(result_dir/'plan.json',declaration)
    print(json.dumps({'stage':'frozen','cases':declaration['cases']}),flush=True)
    outputs=[];fatal=None;started=time.monotonic()
    try:
        import mlx.core as mx
        from keyprint.backends.mlx import MLXModel
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        if shutil.disk_usage(args.output).free < FLOOR: raise OSError('Insufficient disk headroom before model load')
        previous_limit=mx.set_cache_limit(0)
        backend=MLXModel.load(args.model);binding=runtime_binding(max_steps=2048)
        keys=[(Path(plan['confirmation'])/f'owner-{i}.key').read_bytes() for i in range(2)]
        if [sha(k) for k in keys]!=plan['key_commitments']:raise ValueError('Keys differ')
        for original in selected:
            row={'id':original['id']};begin=time.monotonic()
            try:
                if shutil.disk_usage(args.output).free < FLOOR: raise OSError('Insufficient disk headroom before measurement')
                item=source[original['source_index']]
                heads=measure(backend,binding,item['response'],prompt(item))
                values=[score(binding.profile,key,heads) for key in keys]
                old_heads=json.loads((args.study/(row['id']+'.heads.json')).read_text())
                score_path=args.study/(row['id']+'.scores.json')
                if sha(score_path.read_bytes())!=original['scores_sha256']:raise ValueError('Original scores differ')
                old_scores=json.loads(score_path.read_text())
                write(args.output/(row['id']+'.heads.json'),heads);write(args.output/(row['id']+'.scores.json'),values)
                row.update(heads_exact=heads==old_heads,scores_exact=values==old_scores,heads=len(heads['heads']))
                gc.collect();mx.clear_cache()
                row.update(cache_bytes=mx.get_cache_memory(),active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory(),
                    disk_free_bytes=shutil.disk_usage(args.output).free,maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                row['passed']=row['heads_exact'] and row['scores_exact'] and row['cache_bytes']==0 and row['disk_free_bytes']>=FLOOR
            except Exception as exc:
                row.update(passed=False,error={'type':type(exc).__name__,'message':str(exc)})
            row['seconds']=time.monotonic()-begin;outputs.append(row);write(result_dir/(row['id']+'.json'),row)
            print(json.dumps(row),flush=True)
        write(result_dir/'cache-policy.json',{'previous_limit_bytes':previous_limit,'applied_limit_bytes':0})
    except Exception as exc:fatal={'type':type(exc).__name__,'message':str(exc)}
    result={'status':'pass' if len(outputs)==3 and all(r['passed'] for r in outputs) and fatal is None else 'fail',
        'cases':outputs,'fatal':fatal,'seconds':time.monotonic()-started,'new_source_tasks_measured':0,'sdk_policy_changed':False}
    write(result_dir/'summary.json',result);print(json.dumps(result),flush=True)
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
