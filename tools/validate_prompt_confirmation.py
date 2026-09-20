"""Fresh paired Qwen inference for the frozen prompt-aware likelihood rule.

Select twelve new exact-field-disjoint source groups and new keys before work.
Keep all attempts, errors and token caps. A passing screen authorizes further
qualification, never a prompt-free, semantic-quality or deployment claim.
"""
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import time

from prompt_confirmation_selection import history, select, sha, SOURCE_SHA256, SALT
from prompt_conditioned_likelihood import measure
from surrogate_likelihood import score, CHUNK_SIZE, MIXTURE, LOG_CUTOFF
from develop_surrogate_likelihood import write

MAX_TOKENS = 512


def summarize(rows, fatal=None):
    groups = {c:{"attempts":sum(r["condition"]==c for r in rows),
        "available":sum(r["condition"]==c and "error" not in r for r in rows),
        **{f:sum(r.get(f,False) for r in rows if r["condition"]==c) for f in ("matching_flagged","other_flagged")}}
        for c in ("ordinary","marked")}
    short = [r for r in rows if r["condition"]=="marked" and 100<=r.get("words",0)<=400]
    short_hits = sum(r.get("matching_flagged",False) for r in short)
    complete = len(rows)==24 and not fatal and not any("error" in r for r in rows)
    passes = bool(complete and groups["marked"]["matching_flagged"]>=10 and len(short)>=5
        and short_hits>=math.ceil(.8*len(short)) and not any(groups[c][f] for c,f in (
            ("ordinary","matching_flagged"),("ordinary","other_flagged"),("marked","other_flagged"))))
    return {"status":"completed" if complete else "incomplete","attempts":len(rows),
        "errors":sum("error" in r for r in rows),"fatal":fatal,"groups":groups,
        "short_marked":{"count":len(short),"hits":short_hits,"required_hits":math.ceil(.8*len(short)),"minimum_count":5},
        "truncated":sum(r.get("completion")=="length" for r in rows),
        "confirmation_screen_passed":passes,"deployment_calibrated":False,
        "quality_acceptance":False,"prompt_free":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","history-root","development","model","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    raw=args.source.read_bytes()
    if sha(raw)!=SOURCE_SHA256:raise ValueError("Pinned corpus differs")
    development_public=args.development/'public'
    development=json.loads((development_public/'summary.json').read_text())
    audit=json.loads((development_public/'integrity.json').read_text())
    old_plan=json.loads((development_public/'plan.json').read_text())
    if not development['power_gate_passed'] or audit['status']!='pass' or not audit['all_score_terms_recomputed']:
        raise ValueError("Complete audited development pass required")
    if audit['summary_sha256']!=sha((development_public/'summary.json').read_bytes()):
        raise ValueError("Development audit does not bind summary")
    for name,digest in old_plan['dependencies_sha256'].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Frozen detector source differs")
    if (old_plan["fixed_log_cutoff"]!=LOG_CUTOFF or old_plan["mixture"]!=MIXTURE
            or old_plan["chunk_size"]!=CHUNK_SIZE):raise ValueError("Frozen score settings differ")
    manifests,excluded=history(args.history_root)
    tasks,selection=select([json.loads(line) for line in raw.splitlines()],excluded)
    args.output.mkdir(mode=0o700);public=args.output/'public';public.mkdir()
    keys=[secrets.token_bytes(32) for _ in range(2)]
    if len(set(keys))!=2:raise ValueError("Fresh keys must differ")
    previous_text="".join(Path(name).read_text() for name in manifests)
    if any(sha(key) in previous_text for key in keys):raise ValueError("Key commitment already appeared in corpus history")
    for i,key in enumerate(keys):
        with os.fdopen(os.open(args.output/f'owner-{i}.key',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:f.write(key)
    dependencies=("prompt_confirmation_selection.py","prompt_conditioned_likelihood.py","surrogate_likelihood.py",
        "predictability_filter.py","log_tournament.py")
    plan={"scope":__doc__,"script_sha256":sha(Path(__file__).read_bytes()),
        "dependencies_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in dependencies},
        "source":"https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "revision":"bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a","source_sha256":SOURCE_SHA256,
        "attribution":"Copyright (2023) Databricks, Inc.; Wikipedia editors and contributors; CC BY-SA 3.0",
        "license":"https://creativecommons.org/licenses/by-sa/3.0/",
        "prior_manifests_sha256":manifests,"excluded_source_indices":sorted(excluded),
        "selection":selection,"selection_salt":SALT,"tasks":tasks,
        "development_summary_sha256":sha((development_public/'summary.json').read_bytes()),
        "development_audit_sha256":sha((development_public/'integrity.json').read_bytes()),
        "key_commitments":[sha(k) for k in keys],"key_rule":"Task index modulo two; other key is a control",
        "ordering":"Ordinary first for even task index; marked first for odd; independent samples",
        "temperature":.7,"top_k":100,"max_tokens":MAX_TOKENS,"execution":"reference",
        "conditioning":"Original prompt plus original reference context, unedited",
        "chunk_size":CHUNK_SIZE,"mixture":MIXTURE,"fixed_log_cutoff":LOG_CUTOFF,
        "primary":"At least 10/12 matching marked detections and zero ordinary matching, ordinary other or marked other flags",
        "short_screen":"At least five marked responses with 100-400 words; at least 80 percent detected, rounded up",
        "negative_unit":"Ordinary response maximum over two keys is one correlated check; no 36-independent-trials claim",
        "failure_rule":"All 24 attempts retained; no regeneration, replacements, cutoff tuning or discarded truncations",
        "scope_limits":"Fresh to recorded project task groups and keys, not necessarily model training or independent authors/topics"}
    write(public/'plan.json',plan)
    print(json.dumps({"stage":"frozen","selection":selection,"tasks":[{k:t[k] for k in ('source_index','category')} for t in tasks]}),flush=True)
    started=time.monotonic();rows=[];fatal=None
    try:
        from keyprint import Keyprint,sampling
        from keyprint.backends.mlx import ASSETS
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        binding=runtime_binding(max_steps=2048)
        candidates=[Keyprint.from_mlx(args.model,key=key) for key in keys]
        if candidates[0].identity!=candidates[1].identity:raise ValueError("Generation identities differ")
        write(public/'identity.json',{"model_assets":ASSETS,"profile":binding.profile.identity_receipt(),
            "sampling":sampling.identity(),"generation":candidates[0].identity})
        for index,task in enumerate(tasks):
            for condition in (('ordinary','marked') if index%2==0 else ('marked','ordinary')):
                name=f'confirmation-{index:02d}-{condition}'
                row={"id":name,"source_index":task['source_index'],"category":task['category'],"condition":condition,"key_index":index%2}
                begin=time.monotonic()
                try:
                    generated=candidates[index%2].generate(task['prompt'],condition=condition,max_tokens=MAX_TOKENS,output=args.output/name)
                    row.update(text=generated.text,words=len(generated.text.split()),usage=generated.report['usage'],completion=generated.report['payload']['completion'])
                    row['generation_report_sha256']=sha((args.output/name/'report.json').read_bytes())
                    row['generation_journal_sha256']=sha((args.output/name/'journal.jsonl').read_bytes())
                    # Research measurement borrows the already loaded key-blind
                    # model; no generation trace enters the measurement API.
                    heads=measure(candidates[0]._backend,binding,generated.text,task['prompt'])
                    head_path=args.output/(name+'.heads.json');write(head_path,heads)
                    scores=[score(binding.profile,key,heads) for key in keys]
                    score_path=args.output/(name+'.scores.json');write(score_path,scores)
                    row.update(working_log_ratios=[v['working_log_ratio'] for v in scores],
                        matching_flagged=scores[index%2]['flagged'],other_flagged=scores[1-index%2]['flagged'],
                        scored_events=scores[0]['scored_events'],outside_support=scores[0]['outside_support'],
                        heads_sha256=sha(head_path.read_bytes()),scores_sha256=sha(score_path.read_bytes()))
                except Exception as exc:row['error']={"type":type(exc).__name__,"message":str(exc)}
                row['seconds']=time.monotonic()-begin;rows.append(row);write(public/(name+'.json'),row)
                print(json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
    except Exception as exc:fatal={"type":type(exc).__name__,"message":str(exc)}
    summary=summarize(rows,fatal);summary['seconds']=time.monotonic()-started
    write(public/'summary.json',summary);print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
