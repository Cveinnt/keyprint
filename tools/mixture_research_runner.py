"""Dedicated, journaled research caller; never admitted as an SDK execution profile."""
import hashlib
from pathlib import Path
import secrets

import numpy as np

from keyprint._engine.research.keyprint_candidate_v3.adapter import digest
from keyprint._engine.research.keyprint_candidate_v3_caller import DurableJournal


def identity(candidate):
    spec = {'candidate': candidate.identity,
            'research_caller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sdk_execution': False, 'empirical_results_transfer': False}
    return {'schema': 'keyprint.research-mixture-caller.v1',
            'runtime_profile_sha256': digest(spec), 'specification': spec}


def run(candidate, model, prompt, *, key, condition, max_tokens, output,
        backend=None, cache_factory=None, random_bits=secrets.randbits):
    if not prompt or len(prompt) > 8192 or not 1 <= max_tokens <= 1024:
        raise ValueError('Bounded prompt and token cap required')
    if backend is None:
        import mlx.core as backend
    if cache_factory is None:
        from mlx_lm.models.cache import make_prompt_cache
        cache_factory = make_prompt_cache
    output.mkdir(mode=0o700)
    pipe = candidate.pipeline(key, condition=condition, allow_thinking=False, allow_tools=False)
    calls, draws, bits_used = 0, 0, 0
    try:
        with DurableJournal(output / 'journal.jsonl') as journal:
            journal.append({'kind': 'research_start', 'runtime': identity(candidate),
                            'prompt_sha256': digest(prompt), 'condition': condition, 'max_tokens': max_tokens})
            cache = cache_factory(model)

            def forward(ids):
                nonlocal calls
                if calls >= max_tokens + 8:
                    raise RuntimeError('Research model call cap reached')
                calls += 1
                journal.append({'kind': 'model_forward_requested', 'index': calls, 'input_count': len(ids)})
                return model(ids[None], cache=cache)

            def bits(count):
                nonlocal draws, bits_used
                if draws >= 16384 or bits_used + count > 4 * 1024 * 1024:
                    raise RuntimeError('Research random draw budget reached')
                journal.append({'kind': 'random_bits_requested', 'count': count})
                value = random_bits(count)
                draws += 1; bits_used += count
                journal.append({'kind': 'random_bits_returned', 'count': count, 'value_decimal': str(value)})
                return value

            ids = backend.array(prompt, dtype=backend.int32)
            offset = 0
            while len(ids) - offset > 1:
                size = min(2048, len(ids) - offset - 1)
                forward(ids[offset:offset + size])
                backend.eval([entry.state for entry in cache])
                offset += size
            next_input = ids[offset:]
            completion = 'length'
            for index in range(max_tokens):
                raw = forward(next_input)[:, -1, :].astype(backend.float32)
                backend.eval(raw)
                raw = np.array(raw)
                journal.append({'kind': 'prepared_step', 'index': index,
                                'raw_logits_sha256': hashlib.sha256(raw.tobytes()).hexdigest()})
                step = pipe.step(raw, bits)
                journal.append({'kind': 'committed_step', 'index': index, 'token_id': step.token_id})
                if step.stopped is not None:
                    completion = 'eos'
                    break
                next_input = backend.array([step.token_id], dtype=backend.int32)
            if completion == 'length':
                pipe.finish_at_limit(max_tokens)
            receipt = pipe.receipt()
            text = receipt['final']['visible']['text']
            visible_ids = receipt['final']['visible']['token_ids']
            pieces = candidate.reference._base._binding.token_bytes
            rendered = b''.join(pieces[i] for i in visible_ids)
            pending = bytes.fromhex(receipt['final']['visible'].get('pending_utf8_hex', ''))
            if text.encode('utf-8') + pending != rendered:
                raise AssertionError('Research token rendering mismatch')
            result = {'runtime': identity(candidate), 'receipt': receipt, 'text': text,
                      'completion': completion, 'model_forward_calls': calls,
                      'draw_calls': draws, 'exact_rendering': True}
            journal.append({'kind': 'research_terminal', 'completion': completion,
                            'committed_tokens': len(receipt['committed_token_ids'])})
            return result
    finally:
        pipe.close()
