"""Freeze new multilingual source/rubric material and keys before generation.

Preparation only. A manifest is not an executed study, quality acceptance or a
calibrated detector. Original interrupted attempts and quality failures persist.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets
import unicodedata

from audit_paced_study import digest
from balanced_source_session import policy_spec
from paced_key_rank import write
from validate_source_grounded import schedule

LANGUAGES=('English','Spanish','French','Chinese')
DECOYS=199


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC',text).split()).casefold()


def material(records):
    required={'id','language','format','instruction','facts'}
    if (len(records)!=16 or any(set(r)!=required for r in records)
            or len({r['id'] for r in records})!=16
            or Counter(r['language'] for r in records)!=Counter({l:4 for l in LANGUAGES})):
        raise ValueError('Sixteen unique cases, four per declared language, required')
    cases=[];rubrics=[]
    for r in records:
        if (any(type(r[k]) is not str or not r[k].strip() for k in ('id','language','format','instruction'))
                or type(r['facts']) is not list or len(r['facts'])!=6
                or any(type(f) is not str or not f.strip() for f in r['facts'])
                or len(set(r['facts']))!=6):
            raise ValueError('Six distinct explicit facts and nonempty metadata required')
        source='\n'.join(r['facts']);prompt=r['instruction']+'\n\n'+source
        cases.append(dict(id=r['id'],language=r['language'],instruction=r['instruction'],
                          source=source,prompt=prompt,synthetic=True,
                          source_sha256=hashlib.sha256(source.encode()).hexdigest()))
        rubrics.append(dict(id=r['id'],essential_facts=r['facts'],format=r['format'],
                            expected_language=r['language'],
                            qualifiers='Preserve every number, negation, dependency and condition. No unsupported factual additions; ordinary polite connective language is allowed.'))
    if len({normalized(c['source']) for c in cases})!=16 or len({normalized(c['prompt']) for c in cases})!=16:
        raise ValueError('Duplicate source or prompt')
    return cases,rubrics


def protocol():
    return dict(schema='keyprint.balanced-evaluation-protocol.v1',
        scope='Prospective controlled multilingual development study; not natural-distribution, training-held-out, noninferiority or production qualification.',
        provenance='Assistant-authored fictional sources, not external factual assertions or generated output samples.',
        selection='All sixteen declared cases, four fresh reused owner keys, both ordinary and marked conditions: 128 attempts.',
        settings=dict(max_tokens=768,temperature=.7,top_k=100,profile_max_steps=1024,enable_thinking=False,purpose='general'),
        source_policy='Source text is prompt context, not an attached proofreading-protection request. Do not route automatically.',
        failure_policy='Exactly one generation attempt per row. No retries, replacement, rewriting, translation, truncation concealment or favorable-subset summaries. Infrastructure interruption means incomplete, not passed.',
        review=dict(order='Freeze all 128 source-only ratings and uncertainty flags before joining conditions, owner keys or scores.',
                    reviewer='Assistant development review; human ratings, if supplied, remain independent.',
                    fields=['fact_checks','no_unsupported_claims','language_pass','format_pass','uncertain_fields','reason'],
                    facts='Six booleans per output. A fact passes only if all its stated conditions/negations survive. Contradictions or omitted required facts fail; paraphrases preserving meaning may pass.',
                    language='Prose must stay in the requested source language. Names, model identifiers and required technical strings may be retained. Language/script heuristics cannot replace source-only review.',
                    strict='Full-task pass requires every fact, no unsupported claims, requested language and format, successful execution/decoding and EOS. Caps fail full-task acceptance.',
                    sensitivity='Also report accepting every pre-flagged ambiguous failed field. Never invent ambiguity flags after condition/score join.',
                    uncertainty='File freezing records workflow, not proof of what the reviewer previously saw.'),
        detection=dict(statistic='Uniform mean of all eligible keyed layer bits; same canonical context deduplication, no entropy-based selection.',
                       decoys=DECOYS,rank='(1 + number of decoy scores >= owner score) / 200',
                       cutoff='rank <= 0.01; conservative ties; fixed before generation; no threshold search.',
                       denominators='Retain all scheduled rows. Failed, capped, undecodable or unavailable inspections cannot be silently dropped or counted as clean negatives.',
                       views='Report matching key, next-owner-key control and fresh-decoy ranks separately. Never label native-token-path scores as arbitrary-text inspection.',
                       limits='Four owner keys and shared decoys create dependence. Sixty-four ordinary outputs and random-key ranks do not establish a production fixed-key FPR or deployment p-values.'),
        reporting='Report all arms and per-language/component counts plus paired outcomes; preserve old failures. A favorable development screen still needs broader quality, arbitrary-text replay, fresh null calibration, serving and compatibility evidence.',
        preflight='Require completed independently bound balanced native preflight and memory headroom; a loaded model identity or stopped run is insufficient.',
        quality_acceptance=False,detector_calibrated=False,launch_ready=False)


def freeze(source,priors,output):
    records=json.loads(source.read_text());cases,rubrics=material(records)
    previous=set();prior_hashes={}
    for path in priors:
        rows=json.loads(path.read_text());prior_hashes[str(path.resolve())]=digest(path)
        for row in rows:
            for field in ('prompt','source'):
                if row.get(field):previous.add(normalized(row[field]))
    if any(normalized(c[f]) in previous for c in cases for f in ('prompt','source')):
        raise ValueError('New material overlaps an exact normalized prior source or prompt')
    output.mkdir(mode=0o700);private=output/'private';public=output/'public'
    private.mkdir(mode=0o700);public.mkdir()
    keys=[secrets.token_bytes(32) for _ in range(4+DECOYS)]
    if len(set(keys))!=len(keys):raise RuntimeError('Duplicate random key; retain failed bundle')
    for i,key in enumerate(keys[:4]):
        path=private/f'key-{i}';path.write_bytes(key);path.chmod(0o600)
    write(private/'decoys.json',[k.hex() for k in keys[4:]])
    (private/'decoys.json').chmod(0o600)
    write(private/'cases.json',cases);write(private/'rubrics.json',rubrics)
    write(public/'protocol.json',protocol())
    plan=dict(schema='keyprint.balanced-evaluation-bundle.v1',stage='inputs_frozen_no_generation',
              source_file_sha256=digest(source),prior_cases_sha256=prior_hashes,
              overlap_check='NFC/whitespace/casefold exact source and prompt comparison only; not a semantic-independence or training-held-out claim.',
              files_sha256={name:digest(output/name) for name in
                  ('private/cases.json','private/rubrics.json','private/decoys.json','public/protocol.json')},
              key_sha256=[hashlib.sha256(k).hexdigest() for k in keys[:4]],
              decoy_key_sha256=[hashlib.sha256(k).hexdigest() for k in keys[4:]],
              policy=policy_spec(),schedule=schedule(cases),languages={l:4 for l in LANGUAGES},
              total_fact_checks_planned=128*6,generation_executed=False,quality_acceptance=False,
              detector_calibrated=False,launch_ready=False,freezer_sha256=digest(Path(__file__)))
    write(public/'manifest.json',plan)
    validate(output)
    return plan


def validate(root):
    read=lambda p:json.loads(p.read_text())
    m=read(root/'public/manifest.json')
    if m['schema']!='keyprint.balanced-evaluation-bundle.v1' or m['stage']!='inputs_frozen_no_generation':
        raise ValueError('Unknown evaluation manifest')
    expected_files={'private/cases.json','private/rubrics.json','private/decoys.json','public/protocol.json'}
    if (set(m['files_sha256'])!=expected_files or m['languages']!={l:4 for l in LANGUAGES}
            or m['total_fact_checks_planned']!=768 or m['freezer_sha256']!=digest(Path(__file__))
            or any(m[k] is not False for k in ('generation_executed','quality_acceptance','detector_calibrated','launch_ready'))):
        raise ValueError('Frozen manifest scope or preparation state differs')
    for path,h in m['files_sha256'].items():
        if digest(root/path)!=h:raise ValueError('Frozen file changed')
    cases=read(root/'private/cases.json');rubrics=read(root/'private/rubrics.json')
    records=[dict(id=c['id'],language=c['language'],instruction=c['instruction'],format=r['format'],facts=r['essential_facts'])
             for c,r in zip(cases,rubrics,strict=True)]
    expected_cases,expected_rubrics=material(records)
    if (cases!=expected_cases or rubrics!=expected_rubrics or schedule(cases)!=m['schedule']
            or m['policy']!=policy_spec() or read(root/'public/protocol.json')!=protocol()):
        raise ValueError('Frozen material, schedule or policy differs')
    keys=[(root/f'private/key-{i}').read_bytes() for i in range(4)]
    decoys=[bytes.fromhex(h) for h in read(root/'private/decoys.json')]
    if (len(decoys)!=DECOYS or any(len(k)!=32 for k in keys+decoys) or len(set(keys+decoys))!=4+DECOYS
            or [hashlib.sha256(k).hexdigest() for k in keys]!=m['key_sha256']
            or [hashlib.sha256(k).hexdigest() for k in decoys]!=m['decoy_key_sha256']):
        raise ValueError('Frozen owner or decoy keys changed')
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prior-cases',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=freeze(args.source,args.prior_cases,args.output)
    print(json.dumps({'planned_attempts':len(m['schedule']),'fact_checks':m['total_fact_checks_planned'],
                      'languages':m['languages'],'generation_executed':False}))
