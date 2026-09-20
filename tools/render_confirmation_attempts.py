"""Display every confirmation attempt, including failures; never calculate acceptance."""
import argparse
import json
from pathlib import Path

from benchmark_serving import sha
from serving_confirmation import select_cases
from validate_compatibility import screens, write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,required=True)
    args=parser.parse_args()
    public=args.run/'public'
    plan=json.loads((public/'plan.json').read_text())
    cases=select_cases(args.corpus)
    if cases!=plan['cases']:raise ValueError('Original cases differ')
    rows=[]
    for repeat in range(plan['repeats']):
        for case in cases:
            for condition in ('ordinary','marked'):
                name=f"{case['id']}-{repeat}-{condition}"
                row=json.loads((public/(name+'.json')).read_text())
                if row['id']!=name or row['condition']!=condition or row['case']!=case['id']:
                    raise ValueError('Attempt metadata differs')
                if 'error' in row:
                    report=json.loads((args.run/name/'report.json').read_text())['report']
                    row['retained_failure']=report['payload']
                    row['screens']={'complete':False,'nonempty':False,'missing_literals':[],
                                    'semantic_quality':'not_evaluated_due_to_failure'}
                else:
                    if sha(args.run/name/'report.json')!=row['report_sha256']:raise ValueError('Report changed')
                    report=json.loads((args.run/name/'report.json').read_text())['report']
                    if report['rendered_carriers']['visible_text']!=row['text']:raise ValueError('Text differs')
                    row['screens']=screens(case,row['text'],row['completion'])
                rows.append({**row,'case':f"{case['id']}-{repeat}"})
    result={'backend':'Confirmation incomplete: all attempts retained, no accepted timing summary. Frozen Qwen3-8B / MLX; Databricks Dolly, CC BY-SA 3.0.',
            'status':'attempt_display_only','cases':[{**case,'id':f"{case['id']}-{repeat}"} for repeat in range(plan['repeats']) for case in cases],
            'runs':rows,'source':{k:plan[k] for k in ('attribution','source_url','corpus_license','corpus_sha256','corpus_revision')},
            'timing_acceptance':None}
    write_report(args.run,result)
    page=public/'comparison.html'
    page.write_text(page.read_text()+'<footer><p>Instruction/context material: <a href="https://huggingface.co/datasets/databricks/databricks-dolly-15k">Databricks Dolly</a>, <a href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a>. Original instruction plus Context label; outputs newly generated.</p></footer>')
    print(json.dumps({'attempts':len(rows),'errors':sum('error' in r for r in rows),'timing_acceptance':None}))


if __name__=='__main__':main()
