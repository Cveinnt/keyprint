"""Freeze all multilingual source-only ratings before a separate condition join.

Uses the frozen development criteria, not a human/noninferiority acceptance claim.
"""
import argparse
import json
from pathlib import Path
from audit_paced_study import digest
from balanced_study import audit,read
from paced_key_rank import write
from summarize_paced_quality import validate_ratings
from summarize_source_grounded import summarize,diagnostics,METRICS


def helpers():
    return {n:digest(Path(__file__).with_name(n)) for n in
            ('summarize_balanced_quality.py','balanced_study.py','freeze_balanced_evaluation.py',
             'summarize_paced_quality.py','summarize_source_grounded.py')}


def freeze(root,bundle,ratings_path,output):
    checked=audit(root,bundle)
    review=read(root/'private/blind-review.json')
    data=ratings_path.read_bytes();ratings=json.loads(data);validate_ratings(review,ratings)
    output.mkdir(mode=0o700);(output/'ratings.json').write_bytes(data)
    commitment=dict(schema='keyprint.balanced-rating-freeze.v1',evidence=checked,
                    ratings_sha256=digest(output/'ratings.json'),helpers_sha256=helpers(),
                    protocol_sha256=digest(bundle/'public/protocol.json'),
                    reviewer='Assistant source-only development review; human review remains separate',
                    phase='Complete ratings and preflagged ambiguities frozen before this tool joins conditions or scores',
                    limitation='A file commitment cannot prove what a reviewer previously saw.',
                    quality_acceptance=False)
    write(output/'commitment.json',commitment);return commitment


def language_counts(summary,cases):
    result={}
    for language in sorted({c['language'] for c in cases}):
        chosen=[c['id'] for c in cases if c['language']==language]
        result[language]={arm:{metric:sum(summary['by_case'][case][arm][metric] for case in chosen)
                              for metric in ('attempts',*METRICS)} for arm in ('ordinary','marked')}
    return result


def summarize_frozen(root,bundle,frozen):
    checked=audit(root,bundle);commit=read(frozen/'commitment.json')
    if (commit['schema']!='keyprint.balanced-rating-freeze.v1' or commit['evidence']!=checked
            or commit['ratings_sha256']!=digest(frozen/'ratings.json')
            or commit['helpers_sha256']!=helpers() or commit['protocol_sha256']!=digest(bundle/'public/protocol.json')
            or commit['quality_acceptance'] is not False):
        raise ValueError('Frozen ratings, inputs or summary helpers changed')
    review=read(root/'private/blind-review.json');ratings=read(frozen/'ratings.json')
    validate_ratings(review,ratings)
    rows=read(root/'private/runs.json');cases=read(root/'private/cases.json');rubrics=read(root/'private/rubrics.json')
    normalized=[dict(r,error_type=r.get('error_type') or ('DecoderError' if r.get('decode_error') else
                     'SamplerAuditError' if r.get('audit_error') else None)) for r in rows]
    plan=read(root/'public/plan.json')
    strict=summarize(plan,normalized,ratings,rubrics)
    sensitivity=summarize(plan,normalized,ratings,rubrics,accept_uncertain=True)
    return dict(schema='keyprint.balanced-development-quality.v1',commitment_sha256=digest(frozen/'commitment.json'),
                strict=strict,all_uncertain_fields_accepted=sensitivity,
                strict_by_language=language_counts(strict,cases),
                sensitivity_by_language=language_counts(sensitivity,cases),
                uncertain_fields=sum(len(r['uncertain_fields']) for r in ratings),
                raw_token_path_diagnostics=diagnostics(rows),execution=checked,
                scope='Prospectively frozen fictional multilingual development material and assistant review. Correlated cases/keys; not independent human review, powered noninferiority, arbitrary-text detector calibration or universal meaning preservation.',
                quality_acceptance=False,detector_calibrated=False,launch_ready=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','summarize'))
    for n in ('root','bundle','frozen'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--ratings',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.mode=='freeze':
        if args.ratings is None:p.error('--ratings required')
        result=freeze(args.root,args.bundle,args.ratings,args.frozen)
        print(json.dumps({'ratings_sha256':result['ratings_sha256'],'ratings':128,'joined_conditions':False}))
    else:
        if args.output is None:p.error('--output required')
        result=summarize_frozen(args.root,args.bundle,args.frozen);write(args.output,result)
        print(json.dumps({'strict':result['strict']['groups'],'by_language':result['strict_by_language']}))
