"""Actual short-cap SGLang stress test, with every declared attempt retained.

Runs offline in the pinned CPU pilot, not a quality or compatibility benchmark.
It does not force tokens, seed sampling, extend a cap or retry failed outputs.
"""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import sglang as sgl
from transformers import AutoTokenizer

from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.experimental.completion import finalize_completion
from keyprint.experimental.sglang import KeyprintLogitsProcessor
from keyprint.experimental.sglang_runtime import verify_runtime
from validate_portable_utf8 import PROMPTS


def main():
    root = Path('/results')
    public = root / 'public'
    public.mkdir()
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    config = json.loads(Path('/model/config.json').read_text())
    eos = config['eos_token_id']
    binding = ByteLevelBinding.create(tokenizer.backend_tokenizer.to_str(),
        vocabulary_size=config['vocab_size'], special_ids=tokenizer.all_special_ids,
        eos_ids=eos if isinstance(eos, list) else [eos])
    distribution = importlib.metadata.distribution('sglang-cpu')
    identity = verify_runtime(distribution.version, Path(distribution.locate_file('sglang')))
    caps = [1, 2, 4, 8, 16, 32]
    plan = {'prompts': PROMPTS, 'caps': caps, 'conditions': ['ordinary', 'marked'],
            'sampling': 'fresh private draws; no forced tokens or adaptive retries',
            'binding_sha256': binding.digest, 'runtime_identity': identity,
            'sources_sha256': {str(p.relative_to('/keyprint')): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path('/keyprint/src/keyprint/experimental/completion.py'),
                          Path('/keyprint/src/keyprint/experimental/native.py'),
                          Path('/keyprint/src/keyprint/experimental/sglang.py'),
                          Path('/keyprint/src/keyprint/experimental/sglang_runtime.py')]}}
    (public / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    report = {'status': 'started', 'runs': [], 'errors': [], 'quality_acceptance': False}
    engine = None
    try:
        engine = sgl.Engine(model_path='/model', device='cpu', dtype='float32', tp_size=1,
            max_running_requests=2, context_length=512, max_total_tokens=1024,
            disable_overlap_schedule=True, disable_radix_cache=True, disable_cuda_graph=True,
            enable_custom_logit_processor=True, mem_fraction_static=.5)
        for prompt_index, text in enumerate(PROMPTS):
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                tokenize=False, add_generation_prompt=True)
            for cap in caps:
                conditions = ['ordinary', 'marked'] if prompt_index % 2 == 0 else ['marked', 'ordinary']
                params = [{'max_new_tokens': cap, 'temperature': 1., 'top_p': 1., 'top_k': -1,
                           'custom_params': {'keyprint_condition': condition}} for condition in conditions]
                responses = engine.generate([prompt] * 2, params,
                    custom_logit_processor=KeyprintLogitsProcessor.to_str())
                (root / f'host-{prompt_index}-{cap}.json').write_text(json.dumps(responses, indent=2))
                assert len(responses) == 2, 'framework omitted a requested response'
                for condition, response in zip(conditions, responses):
                    row = {'prompt_index': prompt_index, 'max_tokens': cap, 'condition': condition,
                           'host_text': response['text'], 'token_ids': response['output_ids'],
                           'finish_reason': response['meta_info']['finish_reason']['type']}
                    report['runs'].append(row)
                    try:
                        result = finalize_completion(binding, token_ids=row['token_ids'], text=row['host_text'],
                            finish_reason=row['finish_reason'], max_tokens=cap)
                        row.update(text=result.text, completion=result.completion,
                                   carrier_rendering=result.carrier_rendering)
                        ordinary = [i for i in result.token_ids if i not in binding.eos_ids]
                        assert result.text.encode() + result.pending_utf8 == b''.join(binding.pieces[i] for i in ordinary)
                        assert response['meta_info']['completion_tokens'] == len(result.token_ids)
                        row['byte_reconstruction'] = 'pass'
                    except Exception as exc:
                        row['error_type'] = type(exc).__name__
                        report['errors'].append({'prompt_index': prompt_index, 'max_tokens': cap,
                                                 'condition': condition, 'error_type': type(exc).__name__})
                (public / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(prompt_index, cap, 'pair retained', flush=True)
        # Short paths can be identical. Check the complete multiset and each
        # journal's prefixes, without claiming unique request-to-trace matching.
        traced = Counter()
        for path in (root / 'traces').glob('*.jsonl'):
            previous, ids, condition, starts = '0' * 64, [], None, 0
            for sequence, line in enumerate(path.read_bytes().splitlines(keepends=True)):
                row = json.loads(line)
                assert row['sequence'] == sequence and row['previous_sha256'] == previous
                previous = hashlib.sha256(line).hexdigest()
                event = row['event']
                if event['phase'] == 'start':
                    condition = event['condition']
                    starts += 1
                elif event['phase'] == 'prefix_confirmed':
                    assert event['token_ids'] == ids
                elif event['phase'] == 'selected_tentative':
                    ids.append(event['token_id'])
                elif event['phase'] == 'host_prefix_mismatch':
                    raise AssertionError('host prefix mismatch')
            assert starts == 1 and ids
            traced[(condition, tuple(ids))] += 1
        assert traced == Counter((r['condition'], tuple(r['token_ids'])) for r in report['runs'])
        report['journal_multiset_and_prefixes'] = 'pass'
        report['pending_outputs'] = {condition: sum(bool(r.get('carrier_rendering', {}).get('pending_utf8_hex'))
            for r in report['runs'] if r['condition'] == condition) for condition in ('ordinary', 'marked')}
        assert len(report['runs']) == len(PROMPTS) * len(caps) * 2
        assert not report['errors'], 'one or more declared outputs failed validation'
        assert all(report['pending_outputs'].values()), 'no partial suffix observed for both conditions'
        report['status'] = 'pass'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        (public / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if engine is not None:
            engine.shutdown()


if __name__ == '__main__':
    main()
