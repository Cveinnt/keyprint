"""Recheck candidate scores and prefix stability, retaining every audit result."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from predictability_filter import measure, selected_bits
from weighted_null import reference_tail

PREFIX_LENGTHS = (32, 96)
ENTROPY_TOLERANCE = .01


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "study", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root = args.development
    if (root/"public/integrity.json").exists():
        raise ValueError("Never overwrite a prior audit")
    plan = json.loads((root/"public/plan.json").read_text())
    declaration = {"script_sha256":sha(Path(__file__)), "candidate_plan_sha256":sha(root/"public/plan.json"),
                   "prefix_lengths":PREFIX_LENGTHS,"entropy_tolerance_bits":ENTROPY_TOLERANCE,
                   "selection_requirement":"Exactly unchanged for all shared prefix positions",
                   "scope":"Every retained text, two shorter literal prefixes and fresh model caches; not a universal numerical proof"}
    with (root/"public/audit-plan.json").open("x") as stream:
        json.dump(declaration,stream,indent=2)
    audits, scores, error = [], 0, None
    try:
        if plan["script_sha256"] != sha(Path(__file__).with_name("develop_predictability_filter.py")):
            raise ValueError("Development driver changed")
        for name,h in plan["dependencies_sha256"].items():
            if sha(Path(__file__).with_name(name)) != h:
                raise ValueError("Candidate changed: "+name)
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        from keyprint.backends.mlx import MLXModel, ASSETS
        binding = runtime_binding(max_steps=2048)
        if sha(args.study/"public/plan.json")!=plan["source_plan_sha256"]:
            raise ValueError("Source plan changed")
        source_plan = json.loads((args.study/"public/plan.json").read_text())
        keys = [(args.study/f"owner-{i}.key").read_bytes() for i in range(2)]
        if [hashlib.sha256(k).hexdigest() for k in keys]!=source_plan["key_commitments"]:
            raise ValueError("Source keys changed")
        originals = []
        checked_rows = []
        for filename,h in plan["source_rows_sha256"].items():
            source = args.study/"public"/filename
            if sha(source)!=h:
                raise ValueError("Source row changed")
            original = json.loads(source.read_text())
            row = json.loads((root/"public"/filename).read_text())
            if any(row[k]!=original[k] for k in ("id","condition","key_index","words","completion")):
                raise ValueError("Reported source metadata changed")
            path = root/(row["id"]+".selection.json")
            if sha(path)!=row["selection_sha256"]:
                raise ValueError("Private selection changed")
            selection = json.loads(path.read_text())
            for i,key in enumerate(keys):
                bits = selected_bits(binding,original["text"],key,selection)
                tail = reference_tail(bits)["reference_tail"] if len(bits) else 1.
                if tail!=row["weighted"][i]["reference_tail"] or len(bits)!=row["selected_events"]:
                    raise ValueError("Reference score replay differs")
                scores += 1
            index = row["key_index"]
            if (row["matching_flagged"] != (row["weighted"][index]["reference_tail"]<=.005)
                    or row["other_flagged"] != (row["weighted"][1-index]["reference_tail"]<=.005)):
                raise ValueError("Reported hit differs from reference score")
            checked_rows.append(row)
            originals.append((row["id"],selection))
        if scores!=48:
            raise ValueError("All 48 key/text scores required")
        summary = json.loads((root/"public/summary.json").read_text())
        for condition in ("ordinary","marked"):
            for field in ("matching_flagged","other_flagged"):
                if summary["groups"][condition][field]!=sum(r[field] for r in checked_rows if r["condition"]==condition):
                    raise ValueError("Reported group totals differ")
        short = [r for r in checked_rows if r["condition"]=="marked" and 100<=r["words"]<=400]
        if summary["short_marked"]["hits"]!=sum(r["matching_flagged"] for r in short) or summary["short_marked"]["count"]!=len(short):
            raise ValueError("Reported short-answer totals differ")
        backend = MLXModel.load(args.model)
        for name,saved in originals:
            for requested in PREFIX_LENGTHS:
                end = min(requested,len(saved["token_ids"]))
                while end:
                    try:
                        prefix = binding.decode_visible(saved["token_ids"][:end])
                        break
                    except UnicodeDecodeError:
                        end -= 1
                if not end:
                    raise ValueError("No valid UTF-8 prefix")
                fresh = measure(backend,binding,prefix)
                (root/(name+f".prefix-{requested}.json")).write_text(json.dumps(fresh))
                common = 0
                for a,b in zip(saved["token_ids"],fresh["token_ids"]):
                    if a!=b:
                        break
                    common += 1
                if common<max(1,end-8):
                    raise ValueError("Too few shared prefix tokens")
                delta = max(abs(a-b) for a,b in zip(saved["entropy_bits"][:common],fresh["entropy_bits"][:common]))
                unchanged = ([i for i in saved["selected_positions"] if i<common]
                             == [i for i in fresh["selected_positions"] if i<common])
                audit = {"id":name,"requested_prefix_tokens":requested,"common_tokens":common,
                         "max_entropy_difference_bits":delta,"selection_unchanged":unchanged,
                         "identical_raw_heads":saved["raw_head_sha256"][:common]==fresh["raw_head_sha256"][:common],
                         "pass":delta<ENTROPY_TOLERANCE and unchanged}
                audits.append(audit)
                print(json.dumps(audit),flush=True)
    except Exception as exc:
        error = {"type":type(exc).__name__,"message":str(exc)}
    passed = error is None and scores==48 and len(audits)==48 and all(a["pass"] for a in audits)
    result = {"status":"pass" if passed else "failed", "error":error,
              "scores_recomputed":scores,"prefix_checks":audits,
              "candidate_hashes":plan["dependencies_sha256"],
              "candidate_plan_sha256":sha(root/"public/plan.json"),
              "power_summary_sha256":sha(root/"public/summary.json"),
              "dependencies":{p:importlib.metadata.version(p) for p in ("keyprint","mlx","mlx-lm","numpy","scipy")},
              "scope":declaration["scope"]}
    (root/"public/integrity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="prefix_checks"}),flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
