"""Actual OpenAI and Anthropic client requests against one local inference worker.

Every attempt and private receipt is retained. This tests documented text-only
protocol subsets, not hosted GPT/Claude sampling or output-quality acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from keyprint import Keyprint
from keyprint import server
from validate_compatibility import client_check
import validate_compatibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['mlx', 'transformers'], required=True)
    parser.add_argument('--execution', choices=['reference', 'experimental-fast'], default='reference')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.backend != 'mlx' and args.execution != 'reference':
        parser.error('experimental-fast requires MLX')
    if args.backend == 'transformers':
        import torch
        torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    public = args.output / 'public'
    public.mkdir()
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / 'owner.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(key)
    identity = {}
    def loader():
        if args.backend == 'mlx':
            model = Keyprint.from_mlx(args.model, key=key, execution=args.execution)
        else:
            model = Keyprint.from_transformers(args.model, key=key)
        identity.update(model.identity)
        return model
    report = {'status': 'started', 'backend': args.backend, 'execution': args.execution,
              'scope': __doc__, 'hosted_provider_calls': False,
              'versions': {name: importlib.metadata.version(name) for name in ('keyprint', 'openai', 'anthropic')},
              'sources': {Path(name).name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in (__file__, server.__file__, validate_compatibility.__file__)}}
    (public / 'plan.json').write_text(json.dumps(report, indent=2))
    try:
        report['clients'] = client_check(loader, args.output)
        report['status'] = 'pass'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        report['identity'] = identity
        (public / 'clients.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in report.items() if k not in ('identity', 'sources', 'scope')}))


if __name__ == '__main__':
    main()
