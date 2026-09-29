"""Adapt a verified complete candidate packet to the existing private review UI."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_paced_study import audit
from build_source_review import build, replace_once


def prepare(root, output, original):
    checked=audit(root, original)
    if checked['audit_errors']:
        raise ValueError('Resolve sampling audit errors before a factual review; retain original receipts')
    records=json.loads((root/'private/blind-review.json').read_text())
    clean=[]
    for row in records:
        item={k:v for k,v in row.items() if k!='decode_error'}
        if row['decode_error']:
            item['error_type']='Decoder failure: '+row['decode_error']['type']
        item['text']=item['text'] if item['text'] is not None else ''
        clean.append(item)
    source=json.dumps(clean,ensure_ascii=True,indent=2)+'\n'
    page=build(source)
    # Storage remains isolated by this packet's digest; no previous ratings imported.
    page=replace_once(page,'<title>Keyprint · Fidelity review</title>',
                           '<title>Keyprint · paced candidate source review</title>')
    output.mkdir(mode=0o700)
    (output/'review-input.json').write_text(source)
    (output/'index.html').write_text(page)
    receipt={'schema':'keyprint.paced-review-page.v1',
             'original_blind_review_sha256':checked['blind_review_sha256'],
             'adapted_review_sha256':hashlib.sha256(source.encode()).hexdigest(),
             'page_sha256':hashlib.sha256(page.encode()).hexdigest(),
             'rows':len(clean),'ratings':'blank; no human acceptance inferred',
             'adaptation':'Preserve order/all attempts; map decoder failures to visible execution errors; missing text displayed empty',
             'private_source_data':True,'quality_acceptance':False}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--original',type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.root,args.output,args.original)))
