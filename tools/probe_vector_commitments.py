"""Observe real native SDK vectors without changing existing SHA-256 receipts.

This isolated process replaces two module-local hashlib bindings with observers;
the hashlib module and frozen engine are untouched. The observer returns the
original hash object, and additionally tests the developer-only KPV1 codec.
Instrumentation is explicit; this is not an unmodified serving-cost measurement
or qualification of a new SDK receipt schema. Vector encodings remain private.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vector_commitment import encode_vector


def independent_reconstruction(packet):
    """Byte-level decoder, independent of the codec's structured NumPy dtype."""
    assert packet[:4] == b'KPV1' and packet[4:5] in (b'P', b'L')
    width = int.from_bytes(packet[6:10], 'little')
    assert 1 <= width <= 1 << 20
    if packet[5:6] == b'D':
        assert len(packet) == 10 + width * 8
        return packet[10:]
    assert packet[5:6] == b'S' and (len(packet)-10) % 12 == 0
    default = bytes(8) if packet[4:5] == b'P' else bytes.fromhex('000000000000f0ff')
    result = bytearray(default * width)
    previous = -1
    for offset in range(10, len(packet), 12):
        index = int.from_bytes(packet[offset:offset+4], 'little')
        assert previous < index < width
        result[index*8:(index+1)*8] = packet[offset+4:offset+12]
        previous = index
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    public = args.output/'public'; public.mkdir()
    from keyprint import Keyprint
    from keyprint.experimental import fast_mlx, native_mlx
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output/'owner.key', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb') as f:
        f.write(key)
    model = Keyprint.from_mlx(args.model, key=key, execution='experimental-native')
    cases_path = Path(__file__).with_name('inference_cases.json')
    cases = json.loads(cases_path.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan = {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
            'codec_sha256': sha(Path(__file__).with_name('vector_commitment.py')),
            'cases_sha256': sha(cases_path), 'cases': cases, 'identity': model.identity,
            'instrumentation': ['fast_mlx.hashlib binding', 'native_mlx.hashlib binding'],
            'order': 'ordinary first for even case index, marked first for odd',
            'failure_rule': 'retain every attempt; any mismatched bit or receipt fails the probe; no replacements'}
    (public/'plan.json').write_text(json.dumps(plan, indent=2))
    original_fast, original_native = fast_mlx.hashlib, native_mlx.hashlib
    rows, captured, stream = [], [], None
    def observe(kind, size):
        def call(data=b'', **kwargs):
            result = hashlib.sha256(data, **kwargs)
            if type(data) is bytes and len(data) == size:
                values = np.frombuffer(data, dtype='<f8')
                if kind == 'filtered_logits': values = values.reshape(1, -1)
                packet = encode_vector(values, kind)
                assert independent_reconstruction(packet) == data
                entry = {'kind': kind, 'dense_sha256': result.hexdigest(),
                         'encoded_sha256': hashlib.sha256(packet).hexdigest(),
                         'dense_bytes': len(data), 'encoded_bytes': len(packet)}
                captured.append(entry)
                stream.write(json.dumps({**entry, 'packet_hex': packet.hex()})+'\n')
            return result
        return call
    fast_mlx.hashlib = SimpleNamespace(sha256=observe('probability', 151669*8))
    native_mlx.hashlib = SimpleNamespace(sha256=observe('filtered_logits', 151936*8))
    try:
        for index, case in enumerate(cases):
            order = ('ordinary', 'marked') if index % 2 == 0 else ('marked', 'ordinary')
            for condition in order:
                name = case['id']+'-'+condition
                captured = []
                row = {'case': case['id'], 'condition': condition}
                path = args.output/(name+'-vectors.jsonl')
                with os.fdopen(os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as stream:
                    try:
                        result = model.generate(case['prompt'], condition=condition,
                                                max_tokens=case['max_tokens'], output=args.output/name)
                        payload = result.report['payload']
                        probabilities = [v for v in captured if v['kind']=='probability']
                        filtered = [v for v in captured if v['kind']=='filtered_logits']
                        samples = payload['sampling_records']
                        assert len(probabilities) == 2*len(samples)
                        assert len(filtered) == len(samples)
                        for step, sample in enumerate(samples):
                            assert probabilities[step*2]['dense_sha256'] == sample['base_probability_sha256']
                            assert probabilities[step*2+1]['dense_sha256'] == sample['prepared_probability_sha256']
                        # Filter traces are private and not exported in this report.
                        # Their captured bytes are checked, not receipt membership.
                        row.update(status='pass', tokens=len(samples), vectors=len(captured), text=result.text,
                                   completion=payload['completion'],
                                   dense_bytes=sum(v['dense_bytes'] for v in captured),
                                   encoded_bytes=sum(v['encoded_bytes'] for v in captured))
                    except Exception as exc:
                        row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                    stream.flush()
                # Hash only after the buffered file is closed.
                row['vector_file_sha256'] = sha(path)
                rows.append(row)
                (public/'rows.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
                print(json.dumps({k:v for k,v in row.items() if k!='text'}), flush=True)
    finally:
        fast_mlx.hashlib, native_mlx.hashlib = original_fast, original_native
    result = {'status':'pass' if len(rows)==12 and all(r['status']=='pass' for r in rows) else 'failed',
              'outputs':len(rows), 'tokens':sum(r.get('tokens',0) for r in rows),
              'vectors':sum(r.get('vectors',0) for r in rows),
              'dense_bytes':sum(r.get('dense_bytes',0) for r in rows),
              'encoded_bytes':sum(r.get('encoded_bytes',0) for r in rows),
              'scope':'Instrumented actual SDK vectors; independent lossless reconstruction and probability hash reconciliation. Filter trace membership is not checked. No new SDK receipt, cost, semantic or detector acceptance.'}
    (public/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return int(result['status']!='pass')


if __name__ == '__main__': raise SystemExit(main())
