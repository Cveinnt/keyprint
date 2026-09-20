"""Independent byte and randomness reconciliation for visible-only cap reports.

No model inference, new draws, semantic approval or detection claim. Call after
the enclosing audit has checked identities, hashes, prompts and journal chains.
"""
import codecs
import hashlib


def verify_rendering(report, events, token_bytes, eos_token_id):
    payload = report['payload']
    tokens = payload['committed_token_ids']
    completion = payload['completion']
    start, terminal = events[0], events[-1]
    if (start['kind'] != 'response_started' or terminal['kind'] != 'response_terminal'
            or start['purpose'] != 'general' or completion not in ('length', 'eos')
            or terminal['outcome'] != completion):
        raise ValueError('Only completed visible general-generation reports are supported')
    if completion == 'eos':
        if not tokens or tokens[-1] != eos_token_id:
            raise ValueError('EOS completion must retain its selected EOS token')
        visible_ids = tokens[:-1]
    else:
        if len(tokens) != start['max_tokens']:
            raise ValueError('Length completion must reach the exact declared token cap')
        visible_ids = tokens
    if any(type(i) is not int or not 0 <= i < len(token_bytes)
           or token_bytes[i] is None for i in visible_ids):
        raise ValueError('Visible token path contains an invalid ID or control')
    sampled = b''.join(token_bytes[i] for i in visible_ids)
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    text = decoder.decode(sampled, final=False)
    pending = decoder.getstate()[0]
    rendered = report['rendered_carriers']
    expected_status = 'incomplete_utf8_at_token_limit' if pending else 'complete_utf8'
    expected = [{'channel': 'visible', 'status': expected_status,
                 'pending_utf8_hex': pending.hex(), 'committed_token_ids': visible_ids}]
    if (report['carrier_rendering'] != expected or rendered['visible_text'] != text
            or rendered['reasoning_text'] is not None or rendered['tool_texts'] != []
            or rendered['protocol_complete'] is not True
            or text.encode('utf-8') + pending != sampled):
        raise ValueError('Rendered text, retained token IDs or trailing bytes differ')
    diagnostics = payload['literal_diagnostics']
    if len(diagnostics) != 1 or diagnostics[0]['identity']['text_sha256'] != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError('Literal diagnostic is not bound to the rendered text')
    if pending:
        replay = report['literal_replay_status']
        if (completion != 'length' or len(replay) != 1 or replay[0]['channel'] != 'visible'
                or replay[0]['availability'] != 'unavailable' or not replay[0]['reason']
                or diagnostics[0]['availability'] != 'statistic_unavailable'
                or diagnostics[0]['random_key_reference_tail']['value'] is not None
                or any(diagnostics[0][k] is not None for k in ('events', 'ones', 'trials'))):
            raise ValueError('Incomplete UTF-8 carrier must have unavailable literal diagnostics')

    records = payload['sampling_records']
    if len(records) != len(tokens):
        raise ValueError('Sampling record count differs from committed tokens')
    requested = [e for e in events if e['kind'] == 'random_bits_requested']
    returned = [e for e in events if e['kind'] == 'random_bits_returned']
    transcript = []
    expected_order = []
    for index, record in enumerate(records):
        if record['step'] != index or record['token_id'] != tokens[index] or record['channel'] != 'visible':
            raise ValueError('Sampling record is outside the visible token path')
        random = record['randomness']
        draws = random['transcript']
        total = int(random['total_weight_decimal'])
        if total <= 0:
            raise ValueError('Sampling record has no valid integer distribution')
        if not draws and (total != 1 or random['integer_point_decimal'] != '0' or random['token_index'] != 0):
            raise ValueError('Only a unit-mass deterministic sample may omit draws')
        for draw_index, draw in enumerate(draws):
            bits, value = draw['bit_count'], int(draw['value_decimal'])
            accepted = value < total
            if (type(bits) is not int or bits < 1 or bits != (total - 1).bit_length()
                    or not 0 <= value < (1 << bits)
                    or type(draw['accepted']) is not bool or draw['accepted'] != accepted
                    or accepted != (draw_index == len(draws) - 1)):
                raise ValueError('Invalid rejection-sampling transcript')
            transcript.append((index, bits, str(value)))
            expected_order.extend([('random_bits_requested', len(transcript) - 1),
                                   ('random_bits_returned', len(transcript) - 1)])
        if draws and random['integer_point_decimal'] != draws[-1]['value_decimal']:
            raise ValueError('Selected integer differs from the accepted draw')
        expected_order.append(('committed_step', index))
    actual_order = [(e['kind'], e['index']) for e in events
                    if e['kind'] in ('random_bits_requested', 'random_bits_returned', 'committed_step')]
    if actual_order != expected_order:
        raise ValueError('Random draws and token commits are out of order')
    if len(records) != len(tokens) or len(requested) != len(transcript) or len(returned) != len(transcript):
        raise ValueError('Journal and token sampling records have different draw counts')
    for index, (sample, bits, value) in enumerate(transcript):
        request, result = requested[index], returned[index]
        if (request['index'] != index or result['index'] != index
                or request['sample_index'] != sample or result['sample_index'] != sample
                or request['bits'] != bits or result['bits'] != bits or result['value_decimal'] != value):
            raise ValueError('Journal random draws differ from sampling records')
    forwards = [e for e in events if e['kind'] == 'model_forward_reserved']
    if (any(terminal[k] != len(transcript) for k in ('bit_requests', 'bit_values_obtained', 'bit_values_journaled'))
            or terminal['bits_requested'] != sum(t[1] for t in transcript)
            or terminal['model_calls'] != len(forwards)
            or terminal['prefill_calls'] != sum(e['prefill'] is True for e in forwards)
            or terminal['sample_attempts'] != len(tokens)):
        raise ValueError('Consumed-work totals differ from the journal')
    return {'pending_utf8_bytes': len(pending), 'verified_random_draws': len(transcript)}
