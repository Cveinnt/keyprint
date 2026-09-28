"""Five-layer diagnostic on all 500 previously opened public null responses.

Existing heldout keys/data have already been examined in earlier development.
This is an additional rejection screen, never fresh confirmation or calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prefix_candidate import prefix_tail
from layer_likelihood import extract
from validate_null_corpus import SOURCE_SHA256, iid_binomial_upper
from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.source) != SOURCE_SHA256:
        raise ValueError('Pinned source differs')
    original = args.prior / 'public/heldout.jsonl'
    selected = [json.loads(line) for line in original.read_text().splitlines()]
    if len(selected) != 500 or any(row.get('error') for row in selected):
        raise ValueError('Complete 500-response opened study required')
    keys = [(args.prior / f'heldout-{i}.key').read_bytes() for i in range(2)]
    args.output.mkdir(mode=0o700)
    plan = {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
            'candidate_sha256': sha(Path(__file__).with_name('prefix_candidate.py')),
            'source_sha256': SOURCE_SHA256, 'prior_rows_sha256': sha(original),
            'keys_sha256': [hashlib.sha256(k).hexdigest() for k in keys],
            'rule': 'First five layers, exact fair-binomial upper tail <= .005 per key; either of two keys counts once',
            'planned': 500, 'rejection': 'Do not advance a candidate with observed hit fraction above nominal 1%; passing is not fixed-key calibration',
            'selection': 'All 500 previously opened rows, original order, same two keys, no tuning or replacement'}
    save(args.output / 'plan.json', plan)
    corpus = [json.loads(line) for line in args.source.read_bytes().splitlines()]
    binding = runtime_binding(max_steps=2048)
    rows = []
    with (args.output / 'results.jsonl').open('x') as stream:
        for previous in selected:
            row = {k: previous[k] for k in ('source_index', 'text_sha256', 'category', 'words')}
            try:
                text = corpus[row['source_index']]['response']
                if hashlib.sha256(text.encode()).hexdigest() != row['text_sha256']:
                    raise ValueError('Prior text hash differs')
                scores = [prefix_tail(extract(binding, text, key)) for key in keys]
                row.update(scores=scores, any_key_hit=any(s['reference_tail'] <= .005 for s in scores))
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error))
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            if len(rows) % 50 == 0:
                print(f'Null rows {len(rows)}/500', flush=True)
    errors = sum('error_type' in row for row in rows)
    hits = sum(row.get('any_key_hit', False) for row in rows)
    summary = {'rows': len(rows), 'errors': errors, 'hits': hits,
               'screen': 'rejected' if errors or hits > 5 else 'not_rejected_not_calibrated',
               'iid_only_upper_97_5_percent': iid_binomial_upper(hits, 500) if not errors else None,
               'fixed_key_calibrated': False,
               'caveat': 'Opened-data development; iid confidence bound does not address dependencies or fixed-key transfer',
               'results_sha256': sha(args.output / 'results.jsonl')}
    save(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return int(errors > 0)


if __name__ == '__main__':
    raise SystemExit(main())
