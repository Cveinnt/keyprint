"""Observe real reference-path sampling without replacing its probabilities or RNG.

Single-process diagnostic only: temporarily instruments a private engine class.
Intermediate layers retain the original 30-layer PRF domain. They are not results
for separately configured lower-layer profiles or evidence of detector power.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from keyprint._engine.legacy._impl.research import token_source_sparse_execution as sparse
from keyprint._engine.legacy._impl.research.token_source_policy import transform as reference_transform

REVISION = "545dc4251c05440727734bcd94334791f6ab0192"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metrics(probabilities):
    p = probabilities[probabilities > 0]
    return {"entropy_bits": float(-np.sum(p * np.log2(p))),
            "max_probability": float(p.max()),
            "collision_probability": float(np.sum(p * p)),
            "support": len(p)}


def inspect_step(session, q, prepared, repeated):
    p = prepared.probabilities
    decision = session.last_decision
    active = session._condition == "marked" and decision['mode'] != 'startup_ordinary' and not repeated
    protected = decision['protected_token_ids']
    counters = {"branch_roundups": 0, "partition_roundups": 0}
    reference = reference_transform(q, session.profile, session._key, session._context,
                                    protected, counters) if active else q
    if reference.tobytes() != p.tobytes():
        raise AssertionError('Reference and active prepared probabilities differ')
    positive = p > 0
    stages = [{"layer": 0, **metrics(q)}]
    if active:
        ids = [int(i) for i in np.flatnonzero(q > 0)
               if i not in protected and session.profile.classes[i] is not None]
        labels = dict.fromkeys(session.profile.classes[i] for i in ids)
        replay = q.copy()
        if len(labels) >= 2:
            mass = math.fsum(float(q[i]) for i in ids)
            r = tuple(float(q[i]) / mass for i in ids)
            table = sparse.bit_table(session.profile, session._key, session._context, labels)
            for layer in range(session.profile.config.layers):
                r = sparse.update(r, [table[session.profile.classes[i]][layer] for i in ids])
                replay[ids] = [max(sparse.MIN_POSITIVE, mass * weight) for weight in r]
                stages.append({"layer": layer + 1, **metrics(replay)})
        if replay.tobytes() != p.tobytes():
            raise AssertionError('Layer replay differs from active probabilities')
    return {"step": prepared.index, "condition": session._condition,
            "transformed": active, "repeated_context": repeated,
            "base": metrics(q), "prepared": metrics(p), "stages": stages,
            "total_variation": float(.5 * np.sum(np.abs(p - q))),
            "kl_prepared_base_bits": float(np.sum(p[positive] *
                (np.log2(p[positive]) - np.log2(q[positive])))),
            "base_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
            "prepared_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
            "reference_bitwise_equal": True, "layer_replay_bitwise_equal": True}


@contextmanager
def observe(rows):
    """Return original Prepared object; leave draws, commits and engine state intact."""
    cls = sparse.SparseTokenSourceSession
    original_prepare = cls.prepare
    original_commit = cls.commit
    had_own_commit = 'commit' in cls.__dict__
    pending = {}

    def prepare(session, q):
        before = q.tobytes()
        repeated = session._context in session._used
        prepared = original_prepare(session, q)
        row = inspect_step(session, q, prepared, repeated)
        if q.tobytes() != before:
            raise AssertionError('Observer mutated input probabilities')
        rows.append(row)
        pending[id(session)] = (prepared, q.copy(), row)
        return prepared

    def commit(session, prepared, token_id):
        expected, q, row = pending[id(session)]
        if expected is not prepared:
            raise AssertionError('Prepared object changed before commit')
        event = original_commit(session, prepared, token_id)
        row.update(token_id=token_id, selected_base_probability=float(q[token_id]),
                   selected_prepared_probability=float(prepared.probabilities[token_id]))
        del pending[id(session)]
        return event

    cls.prepare, cls.commit = prepare, commit
    try:
        yield
    finally:
        cls.prepare = original_prepare
        if had_own_commit:
            cls.commit = original_commit
        else:
            del cls.commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model.name != REVISION:
        parser.error('Use the registered pinned model snapshot')
    from keyprint import Keyprint
    cases_path = Path(__file__).with_name('prose_quality_cases.json')
    case = next(c for c in json.loads(cases_path.read_text()) if c['id'] == 'french')
    previous = json.loads((args.prior / 'private/runs.json').read_text())
    args.output.mkdir(mode=0o700)
    (args.output / 'private').mkdir(mode=0o700)
    (args.output / 'public').mkdir()
    plan = {"script_sha256": digest(Path(__file__)), "cases_sha256": digest(cases_path),
            "prior_runs_sha256": digest(args.prior / 'private/runs.json'),
            "prior_plan_sha256": digest(args.prior / 'public/plan.json'),
            "model_revision": REVISION, "case": case, "keys": [0, 1, 2, 3],
            "attempts": 8, "max_tokens": 192, "temperature": .7, "top_k": 100,
            "selection": "All four prior key slots, one ordinary and one marked output each; no retries or key selection",
            "ordering": "Key slot ascending; ordinary first for even slots, marked first for odd slots",
            "measurement": "Observer of actual prepare/commit calls; same-prefix before/after probabilities; every layer replayed with original PRF domain",
            "limits": "Diagnostic, not quality acceptance, detector calibration, or lower-layer candidate evaluation. Observer overhead invalidates latency measurements."}
    save(args.output / 'public/plan.json', plan)
    results = []
    for slot in plan['keys']:
        key = (args.prior / f'private/key-{slot}').read_bytes()
        with Keyprint.from_mlx(args.model, key=key, temperature=.7, top_k=100) as wm:
            order = ('ordinary', 'marked') if slot % 2 == 0 else ('marked', 'ordinary')
            for condition in order:
                row = {"key_slot": slot, "condition": condition, "steps": []}
                folder = args.output / 'private' / f'{slot}-{condition}'
                try:
                    with observe(row['steps']):
                        result = wm.generate(case['prompt'], condition=condition,
                                             max_tokens=192, trace=True, output=folder)
                    payload = result.report.get('payload', result.report)
                    committed = [s['token_id'] for s in row['steps']]
                    if committed != payload['committed_token_ids']:
                        raise AssertionError('Observer token IDs do not match actual report')
                    if ''.join(t.text for t in result.trace) != result.text:
                        raise AssertionError('Trace does not reconstruct text')
                    events = [json.loads(line)['event'] for line in (folder / 'journal.jsonl').read_text().splitlines()]
                    draws = [e for e in events if e['kind'] == 'random_bits_returned']
                    report = json.loads((folder / 'report.json').read_text())
                    row.update(text=result.text, completion=payload['completion'],
                        exact_token_trace=True, model_forward_calls=report['reservations']['model_forward'],
                        draw_calls=len(draws), draw_transcript_sha256=hashlib.sha256(
                            json.dumps(draws, sort_keys=True).encode()).hexdigest(),
                        matches_prior_same_condition=sum(r.get('text') == result.text for r in previous
                            if r['key_slot'] == slot and r['case'] == 'french' and r['condition'] == condition))
                except Exception as error:
                    row['error_type'] = type(error).__name__
                    row['error'] = str(error)
                results.append(row)
                save(args.output / 'public/results.json', results)
                print(f"Completed {len(results)}/8; errors: {sum('error_type' in r for r in results)}", flush=True)


if __name__ == '__main__':
    main()
