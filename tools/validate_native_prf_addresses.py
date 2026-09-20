"""Replay native HMAC on addresses from retained real generation token paths.

No new model calls, sampling-distribution replay, SDK promotion or serving claim.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path

from native_prf.batch import BatchSHA256
from keyprint._engine.research.keyprint_candidate_v3.adapter import Candidate
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import pack, replay_events


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    root = args.profile_run
    declaration = json.loads((root/'public/plan.json').read_text())
    summary = json.loads((root/'public/summary.json').read_text())
    candidate = Candidate()
    assert summary['status'] == 'completed'
    assert summary['identity']['score_namespace_sha256'] == candidate.identity['score_namespace_sha256']
    profile = candidate._base._binding.profile
    native = BatchSHA256(args.library)
    key = (root/'owner.key').read_bytes()
    plan = {'script_sha256': sha(Path(__file__)), 'library_sha256': sha(args.library),
            'source_plan_sha256': sha(root/'public/plan.json'),
            'source_summary_sha256': sha(root/'public/summary.json'),
            'profile_digest': profile.digest, 'scope': __doc__}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    rows=[]
    for case in declaration['cases']:
        for condition in ['ordinary','marked']:
            name=f"{case['id']}-{condition}"
            public=json.loads((root/'public'/f'{name}.json').read_text())
            path=root/name
            assert public['condition']==condition and public['report_sha256']==sha(path/'report.json')
            report=json.loads((path/'report.json').read_text())['report']
            assert report['payload']['assigned_condition']==condition
            ids=report['payload']['committed_token_ids']
            previous='0'*64;commits=[]
            for n,line in enumerate((path/'journal.jsonl').read_bytes().splitlines(keepends=True)):
                record=json.loads(line)
                assert record['sequence']==n and record['previous_sha256']==previous
                previous=hashlib.sha256(line).hexdigest()
                if record['event']['kind']=='committed_step': commits.append(record['event']['index'])
            assert commits==list(range(len(ids)))
            assert public['usage']['completion_tokens']==len(ids)
            comparisons=0
            for event in replay_events(profile,key,ids):
                context=pack(event.context)
                prefix=(int(4).to_bytes(4,'big')+len(profile._domain).to_bytes(8,'big')+profile._domain
                        +len(context).to_bytes(8,'big')+context+int(4).to_bytes(8,'big'))
                suffix=len(event.label).to_bytes(8,'big')+event.label
                expected=b''.join(hmac.digest(key,pack([profile._domain,context,i.to_bytes(4,'big'),event.label]),'sha256')
                                  for i in range(profile.config.layers))
                actual=native.digests(key,prefix,[suffix],profile.config.layers)
                assert actual==expected
                if event.eligible:
                    assert tuple(actual[i*32]&1 for i in range(profile.config.layers))==event.bits
                comparisons+=profile.config.layers
            rows.append({'case':name,'retained_tokens':len(ids),'full_digest_comparisons':comparisons,
                         'report_sha256':sha(path/'report.json'),'journal_sha256':sha(path/'journal.jsonl')})
    result={'status':'pass','cases':rows,'outputs':len(rows),
            'tokens':sum(r['retained_tokens'] for r in rows),
            'full_digest_comparisons':sum(r['full_digest_comparisons'] for r in rows),
            'scope':__doc__}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':
    main()
