"""Freeze source-only development ratings, then join conditions in a separate step.

Uses the unchanged earlier source-study metrics. No human approval, calibrated
detection, noninferiority or launch acceptance is inferred from these counts.
"""
import argparse
import json
from pathlib import Path

from audit_paced_study import audit, digest
from summarize_source_grounded import summarize, diagnostics


FIELDS = {'review_id','fact_checks','no_unsupported_claims','language_pass',
          'format_pass','uncertain_fields','reason'}


def validate_ratings(review, ratings):
    """Validate against the blinded packet without joining condition or key."""
    ids = {r['review_id'] for r in review}
    if (len(review) != 128 or len(ids) != 128 or len(ratings) != 128
            or {r.get('review_id') for r in ratings} != ids):
        raise ValueError('Exactly one explicit rating for every blinded attempt required')
    by_id = {r['review_id']:r for r in review}
    for rating in ratings:
        if set(rating) != FIELDS: raise ValueError('Unexpected or missing rating fields')
        row = by_id[rating['review_id']]; facts = rating['fact_checks']
        if not isinstance(facts,list) or len(facts) != len(row['rubric']['essential_facts']):
            raise ValueError('Every source fact needs an explicit rating')
        fields = {'claims':rating['no_unsupported_claims'],'language':rating['language_pass'],
                  'format':rating['format_pass'],**{f'fact:{i}':v for i,v in enumerate(facts)}}
        flags = rating['uncertain_fields']
        if (any(type(v) is not bool for v in fields.values()) or not isinstance(flags,list)
                or any(not isinstance(f,str) for f in flags) or len(set(flags)) != len(flags)
                or any(f not in fields or fields[f] for f in flags)
                or not isinstance(rating['reason'],str) or not rating['reason'].strip()):
            raise ValueError('Explicit booleans, valid uncertainty and a reason required')
        if (row.get('error_type') or row.get('decode_error')) and (any(fields.values()) or flags):
            raise ValueError('Execution/decoder failures cannot be passing or uncertain ratings')


def helper_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in
            ('summarize_paced_quality.py','summarize_source_grounded.py','audit_paced_study.py')}


def freeze(root, original, ratings_path, output):
    checked = audit(root,original)
    if checked['audit_errors']: raise ValueError('Resolve sampling audit errors before factual review')
    review = json.loads((root/'private/blind-review.json').read_text())
    data = ratings_path.read_bytes(); ratings = json.loads(data)
    validate_ratings(review,ratings)
    output.mkdir(mode=0o700)
    (output/'ratings.json').write_bytes(data)
    commitment = {'schema':'keyprint.paced-rating-freeze.v1',
        'reviewer':'assistant development review; independent human ratings remain separate',
        'phase':'Complete source-only ratings frozen before this tool joins conditions or signal counts',
        'protocol_limit':'A file commitment records this workflow; it cannot prove what a reviewer previously saw',
        'ratings_sha256':digest(output/'ratings.json'),'plan_sha256':checked['plan_sha256'],
        'runs_sha256':checked['runs_sha256'],'results_sha256':checked['results_sha256'],
        'blind_review_sha256':checked['blind_review_sha256'],'rubrics_sha256':checked['rubrics_sha256'],
        'cases_sha256':checked['cases_sha256'],'helpers_sha256':helper_hashes(),
        'criteria':'Unchanged original source-study facts, supported claims, language, format and full-task metrics; strict plus all pre-flagged uncertainties accepted',
        'quality_acceptance':False}
    (output/'commitment.json').write_text(json.dumps(commitment,indent=2)+'\n')
    return commitment


def summarize_frozen(root, original, frozen):
    checked = audit(root,original)
    commit = json.loads((frozen/'commitment.json').read_text())
    bound = ('plan_sha256','runs_sha256','results_sha256','blind_review_sha256','rubrics_sha256','cases_sha256')
    if (any(commit[k] != checked[k] for k in bound)
            or commit['ratings_sha256'] != digest(frozen/'ratings.json')
            or commit['helpers_sha256'] != helper_hashes()):
        raise ValueError('Frozen ratings, evidence or summary code changed')
    ratings = json.loads((frozen/'ratings.json').read_text())
    review = json.loads((root/'private/blind-review.json').read_text())
    validate_ratings(review,ratings)
    rows = json.loads((root/'private/runs.json').read_text())
    # The legacy metric counts any execution failure as a failure. Include
    # decoder failures explicitly; do not let a rendering error become a pass.
    normalized = [dict(r, error_type=r.get('error_type') or
                       ('DecoderError' if r.get('decode_error') else None)) for r in rows]
    plan = json.loads((root/'public/plan.json').read_text())
    rubrics = json.loads((root/'private/rubrics.json').read_text())
    return {'schema':'keyprint.paced-development-quality.v1','commitment_sha256':digest(frozen/'commitment.json'),
        'commitment':commit,'strict':summarize(plan,normalized,ratings,rubrics),
        'all_uncertain_fields_accepted':summarize(plan,normalized,ratings,rubrics,accept_uncertain=True),
        'uncertain_fields':sum(len(r['uncertain_fields']) for r in ratings),
        'raw_token_path_counts':diagnostics(rows),
        'execution':{k:checked[k] for k in ('attempts','eos','caps','errors','decode_errors','audit_errors')},
        'scope':'Sixteen reused English source tasks, four reused keys, one experimental profile; descriptive assistant review with correlated attempts. No independent human acceptance, powered noninferiority, detector calibration or universal semantic preservation. Earlier failures retained.',
        'quality_acceptance':False,'detector_calibrated':False,'launch_ready':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('freeze','summarize'));p.add_argument('root',type=Path)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--ratings',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='freeze':
        if args.ratings is None:p.error('--ratings required for freeze')
        result=freeze(args.root,args.original,args.ratings,args.frozen)
        print(json.dumps({'ratings_sha256':result['ratings_sha256'],'ratings':128,'joined_conditions':False}))
    else:
        if args.output is None:p.error('--output required for summarize')
        result=summarize_frozen(args.root,args.original,args.frozen)
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({'groups':result['strict']['groups'],'uncertain_fields':result['uncertain_fields']}))
