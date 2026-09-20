"""Audit fresh-task selection, generation records and all likelihood terms.

Independent graph reachability checks history exclusion. Direct token-mass
likelihood replay checks every score. This is an evidence-integrity audit, not
semantic approval, model-kernel replay or population false-positive calibration.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from audit_prompt_conditioned_likelihood import independent_ratio, half_factor
from audit_serving import verify_output
from analyze_mlx_capacity import read_journal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(row):
    return {(field," ".join(unicodedata.normalize('NFKC',row[field]).casefold().split()))
            for field in ('instruction','context','response') if row[field].strip()}


def validate_groups(rows,plan):
    edges=defaultdict(set)
    for i,row in enumerate(rows):
        for field in fingerprints(row):edges[field].add(i)
    blocked=set(plan['excluded_source_indices']);queue=deque(blocked);visited_fields=set()
    while queue:
        i=queue.popleft()
        for field in fingerprints(rows[i])-visited_fields:
            visited_fields.add(field)
            for j in edges[field]-blocked:
                blocked.add(j);queue.append(j)
    chosen=set()
    for task in plan['tasks']:
        i=task['source_index'];assert i not in blocked and i not in chosen
        assert task['category']==rows[i]['category']
        component={i};queue=deque([i]);seen=set()
        while queue:
            j=queue.popleft()
            for field in fingerprints(rows[j])-seen:
                seen.add(field)
                for k in edges[field]-component:component.add(k);queue.append(k)
        assert component==set(task['group_indices']) and not component & chosen
        assert task['source_record_sha256']==hashlib.sha256(json.dumps(rows[i],sort_keys=True).encode()).hexdigest()
        expected=rows[i]['instruction']+('\n\nReference text:\n'+rows[i]['context'] if rows[i]['context'] else '')
        assert expected==task['prompt'] and hashlib.sha256(expected.encode()).hexdigest()==task['prompt_sha256']
        chosen.update(component)
    return {'excluded_transitive_records':len(blocked),'selected_groups':len(plan['tasks']),'selected_group_records':len(chosen)}


def check_score(profile,key,data,record):
    context,seen,values=(),set(),[];scored=outside=0;maximum_error=0.
    for index,(token,head,term) in enumerate(zip(data['token_ids'],data['heads'],record['terms'],strict=True)):
        support,probabilities=head['support'],head['probabilities']
        assert support==sorted(set(support)) and len(support)==len(probabilities)<=100
        assert all(math.isfinite(p) and p>0 for p in probabilities) and abs(math.fsum(probabilities)-1.)<1e-12
        assert term['index']==index and term['token']==token
        label=profile.classes[token]
        reason='excluded_label' if label is None else 'repeated_context' if context in seen else 'outside_surrogate_support' if token not in support else 'scored'
        assert term['reason']==reason
        base=probabilities[support.index(token)] if token in support else 0.
        assert term['base']==base
        value=0.
        if reason=='scored':
            ratio=independent_ratio(profile,key,context,support,probabilities,token)
            value=half_factor(ratio);scored+=1
            assert abs(math.log(base)+ratio-term['marked_log_probability'])<1e-10
        else:assert term['marked_log_probability'] is None
        outside+=reason=='outside_surrogate_support'
        maximum_error=max(maximum_error,abs(value-term['log_ratio']));assert abs(value-term['log_ratio'])<1e-10
        values.append(value)
        if label is not None:seen.add(context);context=(*context,label)[-profile.config.history:]
    total=math.fsum(values)
    assert abs(total-record['working_log_ratio'])<1e-7 and record['flagged']==(total>=math.log(200.))
    assert record['scored_events']==scored and record['outside_support']==outside and not record['calibrated']
    return {'terms':len(values),'transforms':scored,'term_error':maximum_error,'score_error':abs(total-record['working_log_ratio'])}


def main():
    if not __debug__:raise RuntimeError('Evidence audit requires Python assertions enabled')
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','model'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();public=args.study/'public'
    plan=json.loads((public/'plan.json').read_text());summary=json.loads((public/'summary.json').read_text())
    assert sha(args.source)==plan['source_sha256']
    assert sha(Path(__file__).with_name('validate_prompt_confirmation.py'))==plan['script_sha256']
    for name,digest in plan['dependencies_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    from prompt_confirmation_selection import source_indices,select
    previous=set()
    for name,digest in plan['prior_manifests_sha256'].items():
        assert sha(Path(name))==digest
        previous.update(source_indices(json.loads(Path(name).read_text())))
    assert previous==set(plan['excluded_source_indices'])
    source=[json.loads(line) for line in args.source.read_bytes().splitlines()]
    selected,selection=select(source,previous)
    assert selected==plan['tasks'] and selection==plan['selection']
    group_audit=validate_groups(source,plan)
    from keyprint.backends.mlx import verify_assets,ASSETS
    from keyprint import sampling
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from transformers import AutoTokenizer
    verify_assets(args.model);binding=runtime_binding(max_steps=2048);profile=binding.profile
    tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True,trust_remote_code=False)
    identity=json.loads((public/'identity.json').read_text())
    assert identity['model_assets']==ASSETS and identity['profile']==profile.identity_receipt() and identity['sampling']==sampling.identity()
    keys=[(args.study/f'owner-{i}.key').read_bytes() for i in range(2)]
    assert len(set(keys))==2 and [hashlib.sha256(k).hexdigest() for k in keys]==plan['key_commitments']
    prior_text=''.join(Path(p).read_text() for p in plan['prior_manifests_sha256'])
    assert all(digest not in prior_text for digest in plan['key_commitments'])
    assert plan['fixed_log_cutoff']==math.log(200.) and plan['mixture']==.5 and plan['max_tokens']==512
    totals={'outputs':0,'generation_tokens':0,'heads':0,'scores':0,'terms':0,'transforms':0,'literal_prefix_matches':0}
    maximum_term_error=maximum_score_error=0.;rows=[]
    for index,task in enumerate(plan['tasks']):
        prefix=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        for condition in ('ordinary','marked'):
            name=f'confirmation-{index:02d}-{condition}'
            row=json.loads((public/(name+'.json')).read_text());assert 'error' not in row
            assert row['id']==name and row['source_index']==task['source_index'] and row['condition']==condition and row['key_index']==index%2
            assert row['category']==task['category']
            for filename,field in (('report.json','generation_report_sha256'),('journal.jsonl','generation_journal_sha256')):
                assert sha(args.study/name/filename)==row[field]
            report=json.loads((args.study/name/'report.json').read_text())['report'];events=read_journal(args.study/name/'journal.jsonl')
            augmented={**row,'max_tokens':512,'completion_tokens':row['usage']['completion_tokens'],'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest()}
            totals['generation_tokens']+=verify_output(augmented,report,events,prefix,identity['generation'])
            head_path=args.study/(name+'.heads.json');score_path=args.study/(name+'.scores.json')
            assert sha(head_path)==row['heads_sha256'] and sha(score_path)==row['scores_sha256']
            data=json.loads(head_path.read_text());scores=json.loads(score_path.read_text())
            assert len(scores)==2 and data['token_ids']==list(binding.encode_visible(row['text']))
            assert data['text_sha256']==augmented['text_sha256'] and data['conditioning_prefix_ids']==prefix
            assert data['original_prompt_sha256']==task['prompt_sha256'] and data['original_prompt_used']
            assert not data['private_generation_data_used'] and not data['key_used_for_model_inference']
            totals['literal_prefix_matches']+=data['token_ids']==report['payload']['committed_token_ids'][:len(data['token_ids'])]
            totals['heads']+=len(data['heads'])
            for key,value in zip(keys,scores,strict=True):
                checked=check_score(profile,key,data,value)
                totals['terms']+=checked['terms'];totals['transforms']+=checked['transforms'];totals['scores']+=1
                maximum_term_error=max(maximum_term_error,checked['term_error']);maximum_score_error=max(maximum_score_error,checked['score_error'])
            assert row['working_log_ratios']==[s['working_log_ratio'] for s in scores]
            assert row['matching_flagged']==scores[index%2]['flagged'] and row['other_flagged']==scores[1-index%2]['flagged']
            rows.append(row);totals['outputs']+=1
            print(json.dumps({'audited':name,'totals':totals}),flush=True)
    assert len(rows)==24
    groups={c:{'attempts':12,'available':12,**{f:sum(r[f] for r in rows if r['condition']==c) for f in ('matching_flagged','other_flagged')}} for c in ('ordinary','marked')}
    short=[r for r in rows if r['condition']=='marked' and 100<=r['words']<=400]
    short_hits=sum(r['matching_flagged'] for r in short)
    passed=groups['marked']['matching_flagged']>=10 and len(short)>=5 and short_hits>=math.ceil(.8*len(short)) and not any(groups[c][f] for c,f in (('ordinary','matching_flagged'),('ordinary','other_flagged'),('marked','other_flagged')))
    assert summary['status']=='completed' and summary['attempts']==24 and summary['errors']==0 and summary['fatal'] is None
    assert summary['groups']==groups and summary['short_marked']=={'count':len(short),'hits':short_hits,'required_hits':math.ceil(.8*len(short)),'minimum_count':5}
    assert summary['confirmation_screen_passed']==passed and summary['truncated']==sum(r['completion']=='length' for r in rows)
    result={'status':'pass','scope':__doc__,'group_audit':group_audit,'totals':totals,
        'maximum_term_error':maximum_term_error,'maximum_score_error':maximum_score_error,
        'plan_sha256':sha(public/'plan.json'),'summary_sha256':sha(public/'summary.json'),
        'audit_source_sha256':sha(Path(__file__)),'helpers_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('audit_prompt_conditioned_likelihood.py','audit_serving.py','analyze_mlx_capacity.py')},
        'model_kernels_rerun':False,'deployment_calibrated':False,'sdk_promotion':False}
    with (public/'integrity.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
