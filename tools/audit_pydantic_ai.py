"""Reconcile the installed PydanticAI pilot with wheel, token and journal receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import keyprint
from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_journal(path):
    previous, events = '0' * 64, []
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        value = json.loads(line)
        if value['sequence'] != index or value['previous_sha256'] != previous:
            raise ValueError('Journal hash chain differs')
        previous = hashlib.sha256(line).hexdigest()
        events.append(value['event'])
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--wheel', type=Path, required=True)
    args = parser.parse_args()
    public = args.root / 'public'
    plan = json.loads((public / 'plan.json').read_text())
    result = json.loads((public / 'results.json').read_text())
    if result['status'] != 'pass' or result['server_stopped'] is not True:
        raise ValueError('Complete stopped integration run required')
    if sha(args.wheel) != plan['wheel_sha256']:
        raise ValueError('Wheel hash differs')
    repo = Path(__file__).parents[1]
    if sha(repo / 'tools/validate_pydantic_ai.py') != plan['script_sha256'] or sha(repo / 'examples/pydantic_ai_local.py') != plan['recipe_sha256']:
        raise ValueError('Registered runner or recipe changed')
    installed = Path(keyprint.__file__).parent
    files = []
    with zipfile.ZipFile(args.wheel) as wheel:
        for name in wheel.namelist():
            if not name.startswith('keyprint/') or name.endswith('/'):
                continue
            rel = name.removeprefix('keyprint/')
            if wheel.read(name) != (installed / rel).read_bytes() or wheel.read(name) != (repo / 'src/keyprint' / rel).read_bytes():
                raise ValueError('Wheel, installed package and source differ')
            files.append(name)
    binding = runtime_binding(max_steps=2048)
    paths = sorted(args.root.glob('ordinary-*/report.json')) + sorted((args.root / 'http').glob('*/report.json'))
    if len(paths) != 8:
        raise ValueError('Expected eight actual generation receipts')
    rows = []
    marked_receipts = []
    for path in paths:
        report = json.loads(path.read_text())
        data = report['report']
        payload = data['payload']
        ids = payload['committed_token_ids']
        raw = b''.join(binding.token_bytes[i] or b'' for i in ids)
        text = data['rendered_carriers']['visible_text']
        if raw.decode('utf-8', errors='strict') != text or len(ids) != data['usage']['completion_tokens']:
            raise ValueError('Sampled bytes, rendered text or token count differ')
        events = audit_journal(path.with_name('journal.jsonl'))
        if sum(e['kind'] == 'committed_step' for e in events) != len(ids) or report['reservations']['model_forward'] <= 0:
            raise ValueError('Missing actual model/commit evidence')
        is_marked = path.parent.parent.name == 'http'
        if payload['assigned_condition'] != ('marked' if is_marked else 'ordinary'):
            raise ValueError('Condition differs from intended request')
        if is_marked:
            marked_receipts.append((text, data['usage']))
        rows.append({'name': str(path.relative_to(args.root)), 'condition': payload['assigned_condition'],
                     'tokens': len(ids), 'report_sha256': sha(path), 'journal_sha256': sha(path.with_name('journal.jsonl')),
                     'structured': 'structured_output' in data})
    for row in result['marked']:
        matches = 0
        for text, usage in marked_receipts:
            try:
                equal = json.loads(text) == row['output'] if isinstance(row['output'], dict) else text == row['output']
            except json.JSONDecodeError:
                equal = False
            if equal and usage['completion_tokens'] == row['usage']['output_tokens'] and usage['prompt_tokens'] == row['usage']['input_tokens']:
                matches += 1
        if matches != 1:
            raise ValueError('Client result or usage does not identify a unique receipt')
    audit = {'status': 'pass', 'wheel_sha256': sha(args.wheel), 'package_files_matched': len(files),
             'generations': len(rows), 'committed_tokens': sum(r['tokens'] for r in rows), 'rows': rows,
             'script_sha256': sha(Path(__file__)), 'results_sha256': sha(public / 'results.json'),
             'scope': 'All eight native token renderings and journal chains match. Three Pydantic text outputs match verbatim; one typed output matches parsed JSON values. Input/output usage matches all four client receipts. Hash chains are consistency checks, not authenticity proof. No general quality/detection acceptance.'}
    (public / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('status', 'package_files_matched', 'generations', 'committed_tokens')}))


if __name__ == '__main__':
    main()
