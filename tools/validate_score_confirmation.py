"""Fresh same-key/model confirmation of two previously frozen text-only scores.

Thresholds and weights come from an already completed development comparison.
No tuning, re-ranking, retries, model-probability access or silent exclusions.
Only public/ is exportable. This small study cannot certify deployment error rates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from keyprint import Keyprint
from compare_score_baselines import score
from validate_detection_screen import write_json

PROMPTS = [
    "Explain how a rainbow forms when sunlight interacts with raindrops.",
    "Describe how a small theater group can prepare a community performance.",
    "Explain how a thermos slows the transfer of heat.",
    "Describe how a neighborhood could set up a tool-lending library.",
    "Explain why ocean tides change over the course of a day.",
    "Describe the steps involved in repairing and rebinding an old book.",
    "Explain how mushrooms obtain nutrients and contribute to forest ecosystems.",
    "Describe a careful process for digitizing a collection of family photographs.",
    "En français, explique comment les oiseaux utilisent différentes stratégies pour migrer.",
    "En français, décris comment organiser un échange de vêtements dans un quartier.",
    "En español, explica cómo las raíces ayudan a prevenir la erosión del suelo.",
    "En español, describe cómo organizar un taller de reparación de bicicletas.",
]


def summarize(rows, frozen):
    panels={}
    for name, config in frozen["panels"].items():
        groups={}
        for condition,key in (("ordinary","matching"),("marked","matching"),("marked","other")):
            attempted=[r for r in rows if r["condition"]==condition]
            usable=[r["scores"][key][name] for r in attempted if name in r.get("scores",{}).get(key,{})]
            groups[condition+"_"+key]={"attempts":len(attempted),"available":len(usable),
                "unavailable":len(attempted)-len(usable),"above":sum(v>config["threshold"] for v in usable)}
        panels[name]={"threshold":config["threshold"],"groups":groups}
    return panels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--key",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    frozen=json.loads(args.comparison.read_text())
    expected={"uniform":np.ones(30),"linear_10_to_1":np.linspace(10,1,30)}
    if set(frozen["weights"])!=set(expected) or set(frozen["panels"])!=set(expected):
        raise ValueError("Both fixed baseline scores required")
    for name,w in expected.items():
        if not np.array_equal(w,frozen["weights"][name]) or not math.isfinite(frozen["panels"][name]["threshold"]):
            raise ValueError("Frozen weights or thresholds differ")
    key=args.key.read_bytes()
    if len(key)!=32:raise ValueError("Original owner key required")
    args.output.mkdir(mode=0o700);public=args.output/"public";public.mkdir()
    control=Keyprint.new_key()
    for name,value in (("owner",key),("control",control)):
        with os.fdopen(os.open(args.output/(name+".key"),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as f:
            f.write(value)
    plan={"scope":"Fresh same-generation-key/model small confirmation, not deployment calibration or quality approval",
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "score_source_sha256":hashlib.sha256(Path(__file__).with_name('compare_score_baselines.py').read_bytes()).hexdigest(),
          "comparison_sha256":hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
          "owner_key_commitment":hashlib.sha256(key).hexdigest(),
          "prompts":PROMPTS,"suffix":" Respond in the requested language in about 220 words of continuous prose.",
          "temperature":.7,"top_k":100,"max_tokens":512,"weights":frozen["weights"],
          "thresholds":{n:c["threshold"] for n,c in frozen["panels"].items()},
          "ordering":"ordinary first for even case index, marked first for odd index",
          "comparison":"strict greater-than; all attempts retained; no tuning or retries",
          "other_key":"fresh independent key, fixed for this confirmation"}
    write_json(public/"plan.json",plan)
    rows=[];failure=None
    try:
        from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
        binding=runtime_binding(max_steps=2048)
        candidate=Keyprint.from_mlx(args.model,key=key)
        write_json(public/"identity.json",candidate.identity)
        for index,prompt in enumerate(PROMPTS):
            order=("ordinary","marked") if index%2==0 else ("marked","ordinary")
            for condition in order:
                name=f"confirmation-{index:02d}-{condition}"
                row={"id":name,"condition":condition,"prompt":prompt};started=time.monotonic()
                try:
                    generation=candidate.generate(prompt+plan["suffix"],condition=condition,
                        max_tokens=plan["max_tokens"],output=args.output/name)
                    row.update(text=generation.text,completion=generation.report["payload"]["completion"],scores={})
                    for label,k in (("matching",key),("other",control)):
                        bits=np.asarray([e.bits for e in binding.replay_text(generation.text,k).events if e.eligible])
                        counts=candidate.inspect(generation.text,key=k)
                        if int(bits.sum())!=counts.ones or bits.size!=counts.trials:
                            raise ValueError("Literal event replay and SDK counts disagree")
                        row["scores"][label]={n:score(bits,w) for n,w in expected.items()}
                except Exception as exc:
                    row["error"]=type(exc).__name__
                row["seconds"]=time.monotonic()-started;rows.append(row)
                write_json(public/(name+".json"),row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('text','prompt')}),flush=True)
    except Exception as exc:
        failure=type(exc).__name__
    finally:
        complete=len(rows)==24 and not failure and not any(r.get('error') for r in rows)
        write_json(public/"summary.json",{"status":"completed" if complete else "incomplete",
            "failure":failure,"attempts":len(rows),"panels":summarize(rows,frozen),
            "deployment_calibrated":False,"quality_acceptance":False,"scope":plan["scope"]})
    return 0 if complete else 1


if __name__=="__main__":
    raise SystemExit(main())
