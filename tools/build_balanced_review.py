"""Private multilingual review page with blank independent human ratings."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from balanced_study import audit,read
from build_source_review import build,replace_once


def render(records, *, allow_partial=False):
    clean=[]
    for row in records:
        item={k:v for k,v in row.items() if k!='decode_error'}
        if row.get('decode_error'):item['error_type']='Decoder failure'
        item['text']=item['text'] if item['text'] is not None else ''
        clean.append(item)
    source=json.dumps(clean,ensure_ascii=True,indent=2)+'\n'
    page=build(source,expected_rows=len(records) if allow_partial else 128)
    pattern=r'(<script type="application/json" id="review-data">)(.*?)(</script>)'
    matches=list(re.finditer(pattern,page,re.S))
    if len(matches)!=1:raise ValueError('Review payload template changed')
    match=matches[0];payload=json.loads(match[2]);by_id={r['review_id']:r for r in records}
    for row in payload['rows']:
        original=by_id[row['review_id']]
        language=original['rubric']['expected_language']
        if language!=original['case']['language']:raise ValueError('Review language differs from rubric')
        row['case']['language']=language
    encoded=json.dumps(payload,ensure_ascii=True).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=page[:match.start(2)]+encoded+page[match.end(2):]
    page=replace_once(page,'<title>Keyprint · Fidelity review</title>','<title>Keyprint · Multilingual source review</title>')
    coverage=('All 128 attempts retained.' if len(records)==128 else
              f'INCOMPLETE STUDY: only {len(records)} completed outputs shown out of 128 scheduled attempts. '
              'Interrupted and unstarted attempts remain in the study ledger; this page is not a complete-cohort result.')
    page=replace_once(page,'Private development review. Sources: Databricks Dolly 15k, CC-BY-SA-3.0 dataset and its attributed source material. Sources and derived outputs remain separate from the MIT SDK. This English-only study does not replace prior multilingual failures.',
        'Private development review of fictional source passages in English, Spanish, French and Chinese. '+coverage+
        ' Human ratings start blank and remain separate from assistant judgments. No launch or quality acceptance is implied.')
    page=replace_once(page,"const storageKey='keyprint-source-review:'","const storageKey='keyprint-balanced-source-review:'")
    page=replace_once(page,"row.completion==='length'","(row.completion==='length'||row.completion==='cap')")
    return source,page


def prepare(root,bundle,output):
    checked=audit(root,bundle)
    source,page=render(read(root/'private/blind-review.json'))
    output.mkdir(mode=0o700)
    (output/'review-input.json').write_text(source);(output/'index.html').write_text(page)
    receipt=dict(schema='keyprint.balanced-human-review-page.v1',cohort=checked,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),page_sha256=hashlib.sha256(page.encode()).hexdigest(),
        rows=128,ratings='blank; human judgments not supplied',quality_acceptance=False)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','bundle','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.root,args.bundle,args.output)))
