"""Small condition-blinded task-adherence pilot; no production-quality verdict."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid


def summarize(plan, rows, labels):
    expected = {(c['id'], r, condition) for c in plan['cases']
                for r in range(plan['repeats']) for condition in ('ordinary', 'marked')}
    indexed = {}; by_id = {}
    for row in rows:
        key = row['case'], row['repetition'], row['condition']
        if key not in expected or key in indexed or row['review_id'] in by_id:
            raise ValueError('Duplicate or undeclared attempt')
        indexed[key] = row; by_id[row['review_id']] = row
    ratings = {}
    for label in labels:
        if label['review_id'] not in by_id or label['review_id'] in ratings:
            raise ValueError('Duplicate or unknown review')
        if type(label['task_pass']) is not bool or not label.get('reason'):
            raise ValueError('Review needs boolean task_pass and reason')
        ratings[label['review_id']] = label
    stats = {c: {'attempts': 0, 'task_pass': 0, 'task_fail': 0, 'unreviewed': 0,
                 'runtime_errors': 0} for c in ('ordinary', 'marked')}
    outcomes = {}
    for row in rows:
        stat = stats[row['condition']]; stat['attempts'] += 1
        if row.get('error_type'):
            result = False; stat['runtime_errors'] += 1
        elif row['review_id'] in ratings:
            result = ratings[row['review_id']]['task_pass'] and row['completion'] == 'eos'
        else:
            result = None
        stat['unreviewed' if result is None else 'task_pass' if result else 'task_fail'] += 1
        outcomes[row['review_id']] = result
    pairs = Counter()
    for case in plan['cases']:
        for rep in range(plan['repeats']):
            a = indexed.get((case['id'],rep,'ordinary')); b = indexed.get((case['id'],rep,'marked'))
            if a is None or b is None: continue
            x,y = outcomes[a['review_id']],outcomes[b['review_id']]
            name = 'unreviewed' if x is None or y is None else 'both_pass' if x and y else 'ordinary_only_pass' if x else 'marked_only_pass' if y else 'both_fail'
            pairs[name] += 1
    complete = set(indexed) == expected and all(s['unreviewed']==0 for s in stats.values())
    difference = None
    if complete:
        difference = stats['marked']['task_fail']/stats['marked']['attempts'] - stats['ordinary']['task_fail']/stats['ordinary']['attempts']
    return {'complete': complete, 'conditions': stats, 'paired_outcomes': dict(pairs),
            'marked_minus_ordinary_failure_rate': difference,
            'quality_acceptance': False, 'effect': 'unresolved',
            'scope': 'Descriptive task-adherence pilot, three prompts and one key. Assistant review blinded to condition; not independent human review, a powered noninferiority study or global launch decision.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from keyprint import Keyprint
    cases_path=Path(__file__).with_name('prose_quality_cases.json')
    cases=json.loads(cases_path.read_text()); out=args.output
    out.mkdir(mode=0o700);(out/'public').mkdir();(out/'private').mkdir(mode=0o700)
    plan={'cases':cases,'repeats':4,'max_tokens':192,'temperature':.7,'top_k':100,
          'model_revision':'545dc4251c05440727734bcd94334791f6ab0192',
          'backend':'MLX reference','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'cases_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),
          'ordering':'Alternate first condition by case index plus repetition',
          'failure_rule':'Retain all 24 attempts; no retries, replacement outputs or edits',
          'review_rule':'Freeze assistant task-adherence ratings against supplied rubric before joining condition mapping. Incomplete output fails. Uncertain ratings must be explained, not treated as human acceptance.',
          'primary':'Descriptive ordinary/marked failure counts and paired discordance; no launch pass or causal conclusion from this pilot'}
    (out/'public/plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    key=Keyprint.new_key()
    with os.fdopen(os.open(out/'private/owner.key',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:f.write(key)
    rows=[]
    with Keyprint.from_mlx(args.model,key=key,temperature=.7,top_k=100) as wm:
        for rep in range(4):
            for i,case in enumerate(cases):
                order=('ordinary','marked') if (i+rep)%2==0 else ('marked','ordinary')
                for condition in order:
                    row={'case':case['id'],'repetition':rep,'condition':condition,'review_id':uuid.uuid4().hex[:12]}
                    start=time.perf_counter()
                    try:
                        result=wm.generate(case['prompt'],condition=condition,max_tokens=192,
                            trace=True,output=out/'private'/row['review_id'])
                        payload=result.report.get('payload',result.report)
                        assert ''.join(t.text for t in result.trace)==result.text
                        row.update(text=result.text,completion=payload['completion'],tokens=len(result.trace),exact_rendering=True)
                    except Exception as e:
                        row['error_type']=type(e).__name__
                    row['seconds']=time.perf_counter()-start;rows.append(row)
                    (out/'private/runs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
                    print(f'Completed {len(rows)}/24; runtime errors: {sum("error_type" in r for r in rows)}',flush=True)
    reviews=[{'review_id':r['review_id'],'case':r['case'],'prompt':next(c['prompt'] for c in cases if c['id']==r['case']),
              'rubric':next(c['rubric'] for c in cases if c['id']==r['case']),
              'text':r.get('text',''),'completion':r.get('completion'),
              'word_count':len(r.get('text','').split()),'error_type':r.get('error_type')} for r in rows]
    random.SystemRandom().shuffle(reviews)
    (out/'public/blind-review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2))
    print('Condition-blinded review ready',flush=True)


if __name__=='__main__':main()
