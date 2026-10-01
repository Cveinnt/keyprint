"""Reconcile the full specialized pilot and publish synthetic outputs only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from validate_specialized import sha, save, identities
from specialized_oracle import evaluate


def audit(root, output):
    read=lambda p:json.loads(p.read_text())
    plan=read(root/'public/plan.json');rows=read(root/'private/runs.json')
    commitment=read(root/'public/commitment.json')
    if sha(root/'public/plan.json')!=commitment['plan_sha256'] or identities()!=plan['source_sha256']:
        raise ValueError('Frozen study changed')
    if len(rows)!=32 or len(plan['schedule'])!=32:raise ValueError('Require every attempt')
    continuation=read(root/'public/continuation.json')
    if (sha(root/'private/prefix-runs.json')!=continuation['prefix_runs_sha256'] or
            rows[:16]!=read(root/'private/prefix-runs.json')):raise ValueError('Completed prefix changed')
    cases={c['id']:c for c in plan['cases']};receipts=[]
    for i,(row,expected) in enumerate(zip(rows,plan['schedule'])):
        if row['attempt']!=i or any(row[k]!=v for k,v in expected.items()):raise ValueError('Attempt mismatch')
        if row.get('error_type'):raise ValueError('Retain error and adapt reporting before publication')
        case=cases[row['case']]
        if evaluate(row['text'],case)!=row['oracle']:raise ValueError('Oracle result changed')
        if row['pass'] != (row['exact_trace'] and row['completion']=='eos' and row['oracle']['status']=='pass'):
            raise ValueError('Pass classification changed')
        path=root/'private'/f'attempt-{i:02d}'/'report.json'
        wrapper=read(path);report=wrapper['report'];payload=report['payload']
        if (report['rendered_carriers']['visible_text']!=row['text'] or payload['assigned_condition']!=row['condition']
                or payload['completion']!=row['completion'] or len(payload['committed_token_ids'])!=row['tokens']
                or report['usage']['completion_tokens']!=row['tokens']):raise ValueError('Output/receipt mismatch')
        previous='0'*64;events=[]
        for sequence,line in enumerate(path.with_name('journal.jsonl').read_bytes().splitlines(keepends=True)):
            item=json.loads(line)
            if item['sequence']!=sequence or item['previous_sha256']!=previous:raise ValueError('Journal chain mismatch')
            previous=hashlib.sha256(line).hexdigest();events.append(item['event'])
        committed=[e for e in events if e['kind']=='committed_step']
        if (len(committed)!=row['tokens'] or [e['index'] for e in committed]!=list(range(row['tokens']))
                or events[-1]['kind']!='response_terminal' or events[-1]['outcome']!=row['completion']
                or events[-1]['sampled_tokens']!=row['tokens'] or wrapper['reservations']['model_forward']<=0):
            raise ValueError('Incomplete inference receipt')
        receipts.append({'attempt':i,'report_sha256':sha(path),'journal_sha256':sha(path.with_name('journal.jsonl'))})
    groups={}
    for domain in ['all','python','sql','json','csv']:
        groups[domain]={}
        for arm in ['ordinary','marked']:
            subset=[r for r in rows if r['condition']==arm and (domain=='all' or cases[r['case']]['domain']==domain)]
            groups[domain][arm]={'attempts':len(subset),'passes':sum(r['pass'] for r in subset)}
    paired=Counter();same=0
    for key in range(2):
        for case in cases:
            arms={r['condition']:r for r in rows if r['key_slot']==key and r['case']==case}
            a,b=arms['ordinary'],arms['marked'];same+=a['text']==b['text']
            paired['both_pass' if a['pass'] and b['pass'] else 'ordinary_only' if a['pass'] else
                   'marked_only' if b['pass'] else 'both_fail']+=1
    result={'schema':'keyprint.specialized-results.v1','plan_sha256':commitment['plan_sha256'],
        'runs_sha256':sha(root/'private/runs.json'),'groups':groups,'paired':dict(paired),
        'identical_pairs':same,'tokens':sum(r['tokens'] for r in rows),'runs':rows,'receipts':receipts,
        'quality_acceptance':False,'detector_calibrated':False,
        'scope':'Full frozen pilot. Reconciled returned text, original report, token counts, journal chains and oracle results; no independent model-forward replay, general semantic proof or serving benchmark.'}
    output.mkdir()
    save(output/'results.json',result)
    for name in ['plan.json','commitment.json','continuation.json','preflight-recovery.json']:
        (output/name).write_bytes((root/'public'/name).read_bytes())
    save(output/'resources.json',{name:read(root/name/'result.json') for name in
         ['supervisor','supervisor-clean-assets','supervisor-key1']})
    lines=['# All specialized pilot outputs','', 'Original text, including formatting and casing failures. No rewriting.', '']
    for case in cases.values():
        lines += ['## '+case['id'],'',case['prompt'],'']
        for row in [r for r in rows if r['case']==case['id']]:
            lines += [f"### Key {row['key_slot'] + 1} · {row['condition']} · {'pass' if row['pass'] else 'not passed'}",'',
                      '````text',row['text'],'````','']
    (output/'SAMPLES.md').write_text('\n'.join(lines))
    print(json.dumps({'groups':groups,'paired':dict(paired),'tokens':result['tokens'],'identical_pairs':same}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();audit(a.root,a.output)
