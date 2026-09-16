"""Runnable public-key fixture. No model, API, private key, or detector verdict."""
import argparse,json
import numpy as np
from . import Candidate

def run(draw_index=0):
    candidate=Candidate()
    key=bytes(range(32)) # PUBLIC fixture key. Never use this key for deployment.
    u=(0.15,0.45,0.75,0.95)[draw_index]
    results={}
    for mode in ("ordinary","marked"):
        with candidate.pipeline(key,condition=mode) as pipeline:
            def row(ids,weights=None):
                values=np.full((1,pipeline.model_vocabulary_size),-np.inf,dtype=np.float32)
                values[0,ids]=0. if weights is None else np.log(np.asarray(weights,dtype=np.float64)).astype(np.float32)
                return values
            for token in (785,9104,374):pipeline.step(row([token]),0.5) # The weather is
            selected=pipeline.step(row([11174,11794,9956,30249],[.31,.29,.24,.16]),u)
            pipeline.step(row([151645]),0.5) # Declared EOS
            final=pipeline.finish()
            results[mode]={"selected_text":selected.emitted_text,"text":final.visible.text,"verdict":candidate.score_literal(final.visible.text,key)["verdict"]}
    return {"public_fixture_key":True,"live_model":False,"uniform":u,"results":results}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-index",type=int,choices=range(4),default=0)
    args=parser.parse_args();print(json.dumps(run(args.draw_index),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
