"""Compare two fixed text-only scores on already-opened development artifacts.

The linear 10-to-1 layer weighting follows the documented default in DeepMind's
SynthID Text weighted-mean detector. This code additionally normalizes by the
random-key reference standard deviation; it reports no calibrated probabilities.
Source: https://github.com/google-deepmind/synthid-text/blob/main/src/synthid_text/detector_mean.py
No private model probabilities, prompts or generation journals are used to score.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def score(bits, weights):
    bits=np.asarray(bits,dtype=float);weights=np.asarray(weights,dtype=float)
    if bits.ndim!=2 or not len(bits) or weights.shape!=(bits.shape[1],):
        raise ValueError("Nonempty event-by-layer bits and matching weights required")
    if not np.isin(bits,[0,1]).all() or not np.isfinite(weights).all() or (weights<0).any() or not weights.any():
        raise ValueError("Binary bits and finite nonnegative, nonzero weights required")
    return float(((bits-.5)*weights).sum()/(.5*np.sqrt(len(bits)*(weights**2).sum())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(mode=0o700)
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    binding=runtime_binding(max_steps=2048)
    keys={label:(args.study/(name+".key")).read_bytes() for label,name in (("matching","owner"),("other","control"))}
    weights={"uniform":np.ones(30),"linear_10_to_1":np.linspace(10,1,30)}
    rows=[]
    inputs={}
    for file in sorted((args.study/"public").glob("*.json")):
        if not file.name.startswith(("calibration-","heldout-")):continue
        source=json.loads(file.read_text());inputs[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        row={"id":source["id"],"split":source["split"],"condition":source["condition"],"scores":{}}
        for label,key in keys.items():
            events=binding.replay_text(source["text"],key).events
            bits=np.asarray([e.bits for e in events if e.eligible],dtype=int)
            original=source["scores"][label]
            if int(bits.sum())!=original["ones"] or bits.size!=original["trials"]:
                raise ValueError("Bit extraction differs from original measured counts")
            row["scores"][label]={name:score(bits,w) for name,w in weights.items()}
            if not np.isclose(row["scores"][label]["uniform"],original["statistic"],rtol=1e-12,atol=1e-12):
                raise ValueError("Uniform score replay differs")
        rows.append(row)
    panels={}
    for name in weights:
        calibration=[r["scores"]["matching"][name] for r in rows if r["split"]=="calibration"]
        if len(calibration)!=12:raise ValueError("All original calibration responses required")
        threshold=max(calibration)
        panel={"threshold":threshold,"threshold_rule":"strictly greater than max of original 12 calibration scores","groups":{}}
        for condition,key in (("ordinary","matching"),("marked","matching"),("marked","other")):
            group=[r["scores"][key][name] for r in rows if r["split"]=="heldout" and r["condition"]==condition]
            if len(group)!=12:raise ValueError("All original held-out responses required")
            panel["groups"][condition+"_"+key]={"above":sum(v>threshold for v in group),"total":len(group)}
        panels[name]=panel
    report={"scope":"Opened-data development comparison, not fresh confirmation or detector acceptance",
            "deployment_calibrated":False,"source_files_sha256":inputs,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "weights":{k:v.tolist() for k,v in weights.items()},"rows":rows,"panels":panels,
            "reference":"https://github.com/google-deepmind/synthid-text/blob/main/src/synthid_text/detector_mean.py"}
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(panels,indent=2))


if __name__=="__main__":
    main()
