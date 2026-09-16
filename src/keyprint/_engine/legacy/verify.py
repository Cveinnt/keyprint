"""Recompute all 32 supplied-logit fixtures from the installed package, offline."""
from dataclasses import asdict
from importlib.resources import files
import hashlib,json,sys
import numpy as np
from . import Candidate
PUBLIC_KEY=bytes(range(32))
REPLAY_SHA256="8e7a26cebfb9844199cc1cd58608ccccd84d574ef8e01f67fd98276ae9acd034"

def logits(ids, weights=None):
    row = np.full((1, 151936), -np.inf, dtype=np.float32)
    row[0, ids] = 0. if weights is None else np.log(np.asarray(weights, dtype=np.float64)).astype(np.float32)
    return row

def capture(sdk, spec, condition, uniform):
    observations = []
    with sdk.pipeline(PUBLIC_KEY, condition=condition, purpose=spec['purpose'], source_text=spec['source_text']) as pipeline:
        session = pipeline._raw._visible.session
        prepare = session.prepare
        def observed_prepare(base):
            context = [label.hex() for label in session._context]
            repeated = session._context in session._used
            prepared = prepare(base)
            ids = [int(i) for i in np.flatnonzero(base > 0)]
            decision = session.last_decision
            observations.append({'canonical_context_hex': context, 'repeated_context': repeated,
                'source_policy': decision,
                'candidates': [{'token_id': i, 'text': sdk._scorer.decode_ids([i]) if i != 151645 else '<EOS>',
                    'base_probability': float(base[i]), 'probability': float(prepared.probabilities[i]),
                    'protected': i in decision['protected_token_ids']} for i in ids]})
            return prepared
        # Observe the existing implementation without changing its law or result.
        session.prepare = observed_prepare
        for token in spec['prefix_ids']: pipeline.step(logits([token]), .5)
        selected = pipeline.step(logits(spec['candidate_ids'], spec['weights']), uniform)
        observation = observations[-1]
        pipeline.step(logits([151645]), .5)
        final = pipeline.finish()
        receipt = pipeline.receipt()
        diagnostic = sdk.score_literal(final.visible.text, PUBLIC_KEY)
    return {**observation, 'selected_token_id': selected.token_id,
        'selected_text': selected.emitted_text, 'output_text': final.visible.text,
        'committed_token_ids': list(final.committed_token_ids), 'sdk_receipt': receipt,
        'literal_diagnostic': diagnostic}

def verify():
    raw=files(__package__).joinpath("docs/public-sdk-replay.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REPLAY_SHA256: raise ValueError("public fixture checksum differs")
    saved=json.loads(raw);sdk=Candidate();count=0
    if sdk.identity!=saved["identity"]:raise ValueError("SDK runtime identity differs")
    for case in saved["cases"]:
        for draw in case["draws"]:
            for mode in ("ordinary","marked"):
                result=capture(sdk,case,mode,draw["uniform"])
                if json.loads(json.dumps(result))!=draw[mode]:raise ValueError(f"fixture differs: {case['id']} {mode} {draw['uniform']}")
                count+=1
    if any(n=="mlx" or n.startswith("mlx.") or n=="mlx_lm" for n in sys.modules):raise RuntimeError("unexpected model runtime import")
    return {"status":"pass","condition_replays":count,"runtime":sdk.identity,"fixture_sha256":REPLAY_SHA256,"scope":"Exact supplied-logit records, including probabilities, tokens, rendered text, diagnostic and final receipts. No model generation or empirical quality claim."}
def main():print(json.dumps(verify(),indent=2))
if __name__=="__main__":main()
