"""Declared actual-inference search and exact-prefix replay for a UTF-8 token cap.

The fixed multilingual prompts and seeds are diagnostic stress inputs, not a
quality or rate estimate. All search outputs are retained, including failures.
"""
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import random
from unittest.mock import patch

from keyprint import Keyprint
from keyprint.backends import bytelevel, transformers

PROMPTS = [
    'Continue in Japanese: 庭には小さな木があります。',
    'Continue in Chinese: 春天到了，小树开始生长。',
    'Write a short French explanation of how a seed grows into a tree.',
    'Repeat this exactly without an introduction: 🌱 🌳 🌍',
    'Continue in Korean: 봄이 오면 작은 나무가 자랍니다.',
    'Continue in Arabic: تنمو الشجرة الصغيرة في الحديقة.',
]


def events(directory):
    return [json.loads(line)['event'] for line in (directory / 'journal.jsonl').read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=['ordinary', 'marked'], default='ordinary')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    public = args.output / 'public'
    public.mkdir()
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / 'owner.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
        f.write(key)
    candidate = Keyprint.from_transformers(args.model, key=key)
    plan = {'scope': __doc__, 'identity': candidate.identity, 'prompts': PROMPTS,
            'seeds': list(range(3100, 3100 + len(PROMPTS))), 'condition': args.condition,
            'max_tokens': 96, 'selection': 'first pending-byte token prefix across fixed prompt order',
            'sources': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in (__file__, bytelevel.__file__, transformers.__file__)}}
    (public / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    report = {'status': 'started', 'outputs': [], 'replay': None, 'quality_acceptance': False}
    try:
        for i, prompt in enumerate(PROMPTS):
            directory = args.output / f'search-{i}'
            rng = random.Random(3100 + i)
            with patch('keyprint.backends.transformers.secrets.randbits', rng.getrandbits):
                result = candidate.generate(prompt, condition=args.condition, max_tokens=96, output=directory)
            record = {'index': i, 'text': result.text, 'completion': result.report['completion'],
                      'report_sha256': hashlib.sha256((directory / 'report.json').read_bytes()).hexdigest(),
                      'tokens': result.report['usage']['completion_tokens'], 'partial_cap': None}
            report['outputs'].append(record)
            decoder = codecs.getincrementaldecoder('utf-8')('strict')
            visible = ''
            for j, token in enumerate(result.report['committed_token_ids']):
                piece = candidate._backend.binding.pieces[token]
                if piece is None:
                    break
                visible += decoder.decode(piece, final=False)
                pending = decoder.getstate()[0]
                if pending:
                    record['partial_cap'] = j + 1
                    break
            if record['partial_cap'] is None:
                continue
            replay_dir = args.output / 'replay'
            cap = record['partial_cap']
            rng = random.Random(3100 + i)
            with patch('keyprint.backends.transformers.secrets.randbits', rng.getrandbits):
                replay = candidate.generate(prompt, condition=args.condition, max_tokens=cap, output=replay_dir)
            assert replay.report['completion'] == 'length'
            assert replay.text == visible
            assert replay.report['carrier_rendering'][0]['pending_utf8_hex'] == pending.hex()
            assert replay.report['committed_token_ids'] == result.report['committed_token_ids'][:cap]
            assert replay.report['model_calls'] == replay.report['usage']['completion_tokens'] == cap
            original_events, replay_events = events(directory), events(replay_dir)
            committed = 0
            end = 0
            for end, event in enumerate(original_events):
                committed += event['phase'] == 'committed'
                if committed == cap:
                    break
            # Start identity, actual model-head hashes, transformed weights, every
            # random request/return and selected commit all match in order.
            assert replay_events[:-1] == original_events[:end + 1]
            assert replay_events[-1] == {'phase': 'complete', 'completion': 'length'}
            report['replay'] = {'source_index': i, 'cap': cap, 'text': replay.text,
                                'pending_utf8_hex': pending.hex(), 'exact_prefix_events': len(replay_events) - 1,
                                'extra_model_calls': 0,
                                'report_sha256': hashlib.sha256((replay_dir / 'report.json').read_bytes()).hexdigest()}
            report['status'] = 'pass'
            break
        else:
            report['status'] = 'no_partial_prefix_found'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        (public / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({'status': report['status'], 'search_outputs': len(report['outputs']), 'replay': report['replay']}, ensure_ascii=False))
    if report['status'] != 'pass':
        raise SystemExit('No actual incomplete UTF-8 prefix was verified; qualification remains open')


if __name__ == '__main__':
    main()
