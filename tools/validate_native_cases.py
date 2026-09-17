"""Actual paired inference in the separately pinned offline CPU framework pilots."""
import argparse
import json
import os
from pathlib import Path
import time

from transformers import AutoTokenizer
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile, replay_events
from validate_compatibility import screens, write_report
from check_native_receipts import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['vllm', 'sglang'], required=True)
    args = parser.parse_args()
    root = Path('/results')
    cases = json.loads(Path(__file__).with_name('inference_cases.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    config = json.loads(Path('/model/config.json').read_text())
    eos = config['eos_token_id']; eos = eos if isinstance(eos, list) else [eos]
    binding = ByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(), vocabulary_size=config['vocab_size'],
                                     special_ids=tokenizer.all_special_ids, eos_ids=eos)
    profile = Profile(binding.pieces, tokenizer_identity=binding.digest, config=Config(max_steps=1024))
    key = Path(os.environ['KEYPRINT_KEY_FILE']).read_bytes()
    control = Path('/results/control.key').read_bytes()
    report = {'backend':args.backend + ' CPU pilot', 'cases':cases, 'runs':[], 'clients':{},
              'scope':'Actual paired framework inference; no quality, detector or production acceptance',
              'engineering_failures':[]}
    outputs = []
    if args.backend == 'vllm':
        from vllm import LLM, SamplingParams
        from keyprint.experimental.vllm import KeyprintLogitsProcessor
        engine = LLM(model='/model', dtype='float32', max_model_len=512, enforce_eager=True,
                     max_num_seqs=2, max_num_batched_tokens=512, kv_cache_memory_bytes=256*1024*1024,
                     enable_prefix_caching=False, logits_processors=[KeyprintLogitsProcessor])
    else:
        import sglang as sgl
        from keyprint.experimental.sglang import KeyprintLogitsProcessor
        engine = sgl.Engine(model_path='/model', device='cpu', dtype='float32', tp_size=1,
            max_running_requests=2, context_length=512, max_total_tokens=1024,
            disable_overlap_schedule=True, disable_radix_cache=True, disable_cuda_graph=True,
            enable_custom_logit_processor=True, mem_fraction_static=.5)
    try:
        for index, case in enumerate(cases):
            conditions = ['ordinary','marked'] if index%2 == 0 else ['marked','ordinary']
            started = time.perf_counter()
            if args.backend == 'vllm':
                params = [SamplingParams(max_tokens=case['max_tokens'], temperature=1., top_p=1., top_k=-1,
                          extra_args={'keyprint_condition':condition}) for condition in conditions]
                responses = engine.chat([[{'role':'user','content':case['prompt']}]]*2, params, use_tqdm=False)
                pair = [{'text':r.outputs[0].text, 'token_ids':list(r.outputs[0].token_ids),
                         'completion':'eos' if r.outputs[0].finish_reason == 'stop' else 'length'} for r in responses]
            else:
                prompt = tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],
                                                        tokenize=False, add_generation_prompt=True)
                params = [{'max_new_tokens':case['max_tokens'], 'temperature':1., 'top_p':1., 'top_k':-1,
                           'custom_params':{'keyprint_condition':condition}} for condition in conditions]
                responses = engine.generate([prompt]*2, params, custom_logit_processor=KeyprintLogitsProcessor.to_str())
                pair = [{'text':r['text'], 'token_ids':r['output_ids'],
                         'completion':'eos' if r['meta_info']['finish_reason']['type'] == 'stop' else 'length'} for r in responses]
            elapsed = time.perf_counter()-started
            assert len(pair) == len(conditions), 'framework omitted a requested response'
            for condition, response in zip(conditions, pair):
                outputs.append(response)
                diagnostic = {}
                try:
                    ids = tokenizer.encode(response['text'], add_special_tokens=False)
                    assert binding.render(ids) == response['text']
                    for label, score_key in [('matching',key),('other',control)]:
                        events = replay_events(profile, score_key, ids)
                        bits = [b for event in events if event.eligible for b in (event.bits or ())]
                        diagnostic[label] = {'ones':sum(bits),'trials':len(bits),'fraction':sum(bits)/len(bits) if bits else None}
                except Exception as exc:
                    diagnostic = {'unavailable':type(exc).__name__}
                report['runs'].append({'case':case['id'],'condition':condition,**response,
                    'usage':{'completion_tokens':len(response['token_ids'])},'pair_seconds':elapsed,
                    'screens':screens(case,response['text'],response['completion']),'diagnostic':diagnostic})
                assert response['text'].strip() and 0 < len(response['token_ids']) <= case['max_tokens']
            (root/'outputs.json').write_text(json.dumps(outputs,indent=2))
            write_report(root,report)
            print(case['id'], 'pair complete', flush=True)
        report['token_path_verification'] = check(root)
    except Exception as exc:
        report['engineering_failures'].append(type(exc).__name__)
        write_report(root,report)
        raise
    finally:
        if args.backend == 'sglang':engine.shutdown()
    write_report(root,report)


if __name__ == '__main__':
    main()
