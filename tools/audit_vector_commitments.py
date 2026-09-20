"""Offline audit of retained KPV1 observations, without new inference.

Checks canonical packets, reconstructed dense hashes, exported probability
hashes and existing generation journals. Private filter-trace membership is
not checked. A successful audit does not turn a failed probe into a passing run.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_mlx_capacity import read_journal
from audit_serving import verify_output
from vector_commitment import decode_vector, encode_vector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_vectors(entries, samples):
    if not samples or len(entries) != 3 * len(samples):
        raise ValueError('Expected three observed vectors per committed step')
    dense_bytes = encoded_bytes = 0
    for index, entry in enumerate(entries):
        slot = index % 3
        kind = 'filtered_logits' if slot == 0 else 'probability'
        width = 151936 if slot == 0 else 151669
        packet = bytes.fromhex(entry['packet_hex'])
        decoded = decode_vector(packet)
        dense = decoded.values.tobytes()
        digest = hashlib.sha256(dense).hexdigest()
        if (entry['kind'] != kind or decoded.kind != kind or decoded.values.size != width
                or entry['dense_bytes'] != len(dense) or entry['encoded_bytes'] != len(packet)
                or entry['dense_sha256'] != digest
                or entry['encoded_sha256'] != hashlib.sha256(packet).hexdigest()
                or encode_vector(decoded.values, kind) != packet):
            raise ValueError('Vector bytes, kind, width or commitment differs')
        if slot:
            field = 'base_probability_sha256' if slot == 1 else 'prepared_probability_sha256'
            if samples[index // 3][field] != digest:
                raise ValueError('Exported probability hash differs')
        dense_bytes += len(dense)
        encoded_bytes += len(packet)
    return {'vectors': len(entries), 'dense_bytes': dense_bytes, 'encoded_bytes': encoded_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--probe-source', type=Path, required=True)
    args = parser.parse_args()
    root, public = args.run, args.run / 'public'
    destination = public / 'vector-audit.json'
    if destination.exists():
        raise ValueError('Never overwrite an existing audit')
    plan = json.loads((public / 'plan.json').read_text())
    source_rows = json.loads((public / 'rows.json').read_text())
    initial_summary = json.loads((public / 'summary.json').read_text())
    if (sha(args.probe_source) != plan['script_sha256']
            or sha(Path(__file__).with_name('vector_commitment.py')) != plan['codec_sha256']
            or sha(Path(__file__).with_name('inference_cases.json')) != plan['cases_sha256']):
        raise ValueError('Frozen probe, codec or prompts differ')
    from keyprint.backends.mlx import ASSETS
    from transformers import AutoTokenizer
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if sha(args.model / name) != ASSETS[name]:
            raise ValueError('Pinned tokenizer differs')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    expected = [(case, condition) for i, case in enumerate(plan['cases'])
                for condition in (('ordinary', 'marked') if i % 2 == 0 else ('marked', 'ordinary'))]
    if len(source_rows) != len(expected) or len(expected) != 12:
        raise ValueError('Missing or extra attempts')
    rows = []
    for original, (case, condition) in zip(source_rows, expected):
        if (original['case'], original['condition']) != (case['id'], condition):
            raise ValueError('Attempt order differs')
        name = case['id'] + '-' + condition
        vectors = root / (name + '-vectors.jsonl')
        if sha(vectors) != original['vector_file_sha256']:
            raise ValueError('Captured vector file differs')
        entries = [json.loads(line) for line in vectors.read_text().splitlines()]
        report_path, journal_path = root / name / 'report.json', root / name / 'journal.jsonl'
        report = json.loads(report_path.read_text())['report']
        payload = report['payload']
        row = {'case': case['id'], 'condition': condition, 'max_tokens': case['max_tokens'],
               'completion_tokens': len(payload['committed_token_ids']), 'completion': payload['completion'],
               'text': report['rendered_carriers']['visible_text']}
        row['text_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
        prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': case['prompt']}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        verify_output(row, report, read_journal(journal_path), prompt_ids, plan['identity'])
        row.update(check_vectors(entries, payload['sampling_records']))
        row.update(report_sha256=sha(report_path), journal_sha256=sha(journal_path),
                   vector_file_sha256=sha(vectors), original_probe_status=original['status'])
        rows.append(row)
    result = {'status': 'pass', 'scope': __doc__, 'original_probe_status': initial_summary['status'],
              'outputs': len(rows), 'tokens': sum(r['completion_tokens'] for r in rows),
              **{field: sum(r[field] for r in rows) for field in ('vectors', 'dense_bytes', 'encoded_bytes')},
              'audit_sha256': sha(Path(__file__)), 'rows': rows}
    with destination.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'scope')}))


if __name__ == '__main__':
    main()
