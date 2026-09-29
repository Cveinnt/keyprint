"""Research inference loop: native heads -> exact integer draws -> retained tokens.

Callbacks supply native forwards, random bits and unmodified tokenizer decoding.
No generation retries, output repair, automatic routing or quality acceptance.
"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

from paced_source_session import PacedSourceSession, SamplingFailure, SourceRequest
from replay_wide_mlx import ordinary_weights


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def generate(*, profile, binding, key, condition, prompt_ids, forward, decode,
             random_bits, output, max_tokens=768, temperature=.7, top_k=100,
             request=SourceRequest()):
    if (type(max_tokens) is not int or not 1 <= max_tokens <= profile.config.max_steps
            or not prompt_ids or any(type(i) is not int or not 0 <= i < len(binding.pieces) for i in prompt_ids)
            or not 0 < temperature <= 2 or type(top_k) is not int or not 1 <= top_k <= 4096):
        raise ValueError('Invalid bounded inference settings or prompt IDs')
    # Do not accept a tokenizer/binding that silently changes the profile mapping.
    if (len(binding.pieces) != len(profile.classes) or set(binding.eos_ids) != profile.eos_ids
            or tuple((p.translate(None, b' \t\r\n\f\v') or None) if p is not None else None
                     for p in binding.pieces) != profile.classes):
        raise ValueError('Binding differs from declared profile')
    output = Path(output); output.mkdir(mode=0o700)
    session = PacedSourceSession(profile, key, condition=condition, request=request)
    tokens = []; inputs = list(prompt_ids); began = time.monotonic()
    result = {'schema': 'keyprint.paced-native-output.v1', 'condition': condition,
              'profile_sha256': profile.digest, 'completion': 'incomplete', 'model_calls': 0,
              'quality_acceptance': False, 'detector_calibrated': False}
    phase = 'start'
    with (output / 'journal.jsonl').open('x') as journal:
        def emit(event):
            journal.write(json.dumps(event, separators=(',', ':')) + '\n'); journal.flush()
        emit({'phase': 'start', 'prompt_token_ids': prompt_ids,
              'profile_sha256': profile.digest, 'max_tokens': max_tokens,
              'temperature': temperature, 'top_k': top_k})
        try:
            for index in range(max_tokens):
                phase = 'forward'; result['model_calls'] += 1
                raw = forward(inputs)
                phase = 'filter'
                base = ordinary_weights(raw, binding, temperature=temperature, top_k=top_k)
                phase = 'prepare'; step = session.prepare(base)
                emit({'phase': 'prepared', 'index': index,
                      'raw_logits_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                      'base_weights_sha256': hashlib.sha256(base.tobytes()).hexdigest(),
                      'token_ids': step.token_ids,
                      'base_hex': [float(base[i]).hex() for i in step.token_ids],
                      'distribution': asdict(step.distribution), 'decision': session.last_decision})
                phase = 'draw'; draw = session.draw(step, random_bits)
                emit({'phase': 'drawn', 'index': index, 'draw': asdict(draw)})
                phase = 'commit'; session.commit(step, draw.token_index)
                tokens.append(draw.token_index)
                emit({'phase': 'committed', 'index': index, 'token_id': draw.token_index})
                if draw.token_index in profile.eos_ids:
                    result['completion'] = 'eos'; break
                inputs = [draw.token_index]
            else:
                result['completion'] = 'cap'
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error), failed_phase=phase)
            if isinstance(error, SamplingFailure):
                result['failed_draw'] = {'transcript': [asdict(t) for t in error.transcript],
                                        'callback_calls': error.callback_calls}
            emit({'phase': 'failure', **result})
        finally:
            # The frozen session's close is intentionally not idempotent;
            # EOS and fail-closed numerical/entropy errors already close it.
            if not session._closed: session.close()
        result.update(committed_token_ids=tokens, source_receipt=session.source_receipt(),
                      elapsed_seconds=time.monotonic() - began)
        try:
            # EOS is not user-facing text. No other token, character or language edit.
            visible = [i for i in tokens if i not in profile.eos_ids]
            rendered = decode(visible)
            if not isinstance(rendered, str): raise TypeError('Decoder must return text')
            raw_text = b''.join(binding.pieces[i] for i in visible).decode('utf-8', errors='strict')
            if rendered != raw_text:
                result['rejected_decoder_text'] = rendered
                raise ValueError('Decoder changed committed token bytes')
            result['text'] = rendered
        except Exception as error:
            result['decode_error'] = {'type': type(error).__name__, 'message': str(error)}
        emit({'phase': 'finished', 'completion': result['completion'], 'tokens': len(tokens)})
    result['journal_sha256'] = hashlib.sha256((output / 'journal.jsonl').read_bytes()).hexdigest()
    save(output / 'result.json', result)
    return result


def audit_output(output, profile, key, *, binding, request=SourceRequest()):
    """Replay sampler/lifecycle and independently check rational CDF and mass.

    Does not replay native model heads or validate semantic quality. Refuses
    failed/abruptly interrupted attempts; their receipts remain available.
    """
    import numpy as np
    output = Path(output)
    result = json.loads((output / 'result.json').read_text())
    if (result['profile_sha256'] != profile.digest or result['completion'] not in ('eos', 'cap')
            or 'error_type' in result or 'decode_error' in result
            or hashlib.sha256((output / 'journal.jsonl').read_bytes()).hexdigest() != result['journal_sha256']):
        raise ValueError('Incomplete, failed or changed inference receipt')
    session = PacedSourceSession(profile, key, condition=result['condition'], request=request)
    committed = []; pending = None; drawn = None; state = 'start'; maximum = None
    with (output / 'journal.jsonl').open() as f:
        for line in f:
            e = json.loads(line)
            if e['phase'] == 'start' and state == 'start':
                if e['profile_sha256'] != profile.digest: raise ValueError('Journal profile changed')
                maximum = e['max_tokens']; state = 'prepared'
            elif e['phase'] == 'prepared' and state == 'prepared':
                if e['index'] != len(committed): raise ValueError('Step order changed')
                ids = e['token_ids']; q = np.zeros(len(profile.classes), dtype=np.float64)
                if ids != sorted(set(ids)) or len(ids) != len(e['base_hex']): raise ValueError('Invalid support')
                for i, h in zip(ids, e['base_hex']):
                    if type(i) is not int or not 0 <= i < len(q): raise ValueError('Invalid token')
                    q[i] = float.fromhex(h)
                    if not np.isfinite(q[i]) or q[i] <= 0: raise ValueError('Invalid base weight')
                if hashlib.sha256(q.tobytes()).hexdigest() != e['base_weights_sha256']:
                    raise ValueError('Base weights changed')
                pending = session.prepare(q)
                if (list(pending.token_ids) != ids or json.loads(json.dumps(asdict(pending.distribution))) != e['distribution']
                        or json.loads(json.dumps(session.last_decision)) != e['decision']):
                    raise ValueError('Prepared distribution or source decision differs')
                base = [Fraction(float(q[i])) for i in ids]; mass = sum(base)
                actual = [Fraction(w, pending.distribution.total) for w in pending.distribution.weights]
                protected = set(e['decision']['protected_token_ids'])
                for i, b, a in zip(ids, base, actual):
                    b /= mass
                    if not b/2 <= a <= 2*b: raise ValueError('Global ratio violation')
                    if (profile.classes[i] is None or i in protected) and a != b:
                        raise ValueError('Excluded probability changed')
                state = 'drawn'
            elif e['phase'] == 'drawn' and state == 'drawn':
                if e['index'] != len(committed): raise ValueError('Draw order changed')
                entries = iter(e['draw']['transcript'])
                def bits(count):
                    entry = next(entries)
                    if entry['bit_count'] != count: raise ValueError('Bit request differs')
                    return entry['value']
                drawn = session.draw(pending, bits)
                if next(entries, None) is not None or json.loads(json.dumps(asdict(drawn))) != e['draw']:
                    raise ValueError('Draw transcript differs')
                point = Fraction(drawn.integer_point, drawn.total_weight); cumulative = Fraction(0)
                for i, w in zip(pending.token_ids, pending.distribution.weights):
                    cumulative += Fraction(w, pending.distribution.total)
                    if point < cumulative:
                        if i != drawn.token_index: raise ValueError('Independent CDF differs')
                        break
                state = 'committed'
            elif e['phase'] == 'committed' and state == 'committed':
                if e['index'] != len(committed) or e['token_id'] != drawn.token_index:
                    raise ValueError('Commit differs')
                session.commit(pending, e['token_id']); committed.append(e['token_id'])
                pending = drawn = None; state = 'prepared'
            elif e['phase'] == 'finished' and state == 'prepared':
                if e['tokens'] != len(committed) or e['completion'] != result['completion']:
                    raise ValueError('Final journal counts differ')
                state = 'finished'
            else:
                raise ValueError('Invalid journal lifecycle')
    if not session._closed: session.close()
    if (state != 'finished' or committed != result['committed_token_ids']
            or result['model_calls'] != len(committed) or session.source_receipt() != result['source_receipt']
            or not committed or len(committed) > maximum
            or (result['completion'] == 'eos') != (committed[-1] in profile.eos_ids)
            or (result['completion'] == 'cap' and len(committed) != maximum)):
        raise ValueError('Output counts, ending or source receipt differs')
    expected_text = b''.join(binding.pieces[i] for i in committed if i not in profile.eos_ids).decode('utf-8', errors='strict')
    if result['text'] != expected_text: raise ValueError('Rendered text differs from committed bytes')
    return {'verified_draws': len(committed), 'native_heads_replayed': False, 'quality_acceptance': False}
