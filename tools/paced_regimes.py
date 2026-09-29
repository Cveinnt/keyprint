"""Descriptive entropy regimes of every saved candidate path; no inference.

Ordinary rows describe actual ordinary generation, never marked counterfactuals.
Marked conditional metrics must reconcile the previous complete information run.
Bins are descriptive, not a proposed detector or generation selection rule.
"""
import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path

from audit_paced_study import audit, digest
from paced_information import metrics, same_distribution
from paced_source_session import integer_distribution, IntegerDistribution
from paced_key_rank import write

CUTS = (.1, .5, 1., 2.)
LABELS = ('[0,0.1)', '[0.1,0.5)', '[0.5,1)', '[1,2)', '[2,infinity)')


def entropy_bin(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative entropy required')
    return LABELS[bisect_right(CUTS, value)]


def blank():
    return dict(steps=0, kl_sum=0., tv_sum=0., entropy_sum=0.,
                boundary_freezes=0, executed_layers=0, steps_with_freezes=0,
                max_base_ge_99pct=0, eos_selected=0)


def inspect(path, row, eos_ids):
    if digest(path) != row['journal_sha256']:
        raise ValueError('Journal hash changed')
    groups = {name: blank() for name in LABELS}
    tokens, pending, path_metrics = [], None, []
    with path.open() as f:
        for line in f:
            e = json.loads(line)
            if e['phase'] == 'prepared':
                if pending is not None or e['index'] != len(tokens):
                    raise ValueError('Invalid prepared order')
                decision = e['decision']
                if decision['mode'] != 'full_mark' or decision['protected_token_ids']:
                    raise ValueError('Only original general-purpose mode admitted')
                ids = e['token_ids']
                raw = tuple(float.fromhex(h) for h in e['base_hex'])
                if ids != sorted(set(ids)) or len(ids) != len(raw) or any(
                        not math.isfinite(v) or v <= 0 for v in raw):
                    raise ValueError('Invalid sparse base support')
                base = integer_distribution(raw)
                actual = IntegerDistribution(tuple(e['distribution']['weights']),
                                             e['distribution']['total'])
                if row['condition'] == 'ordinary' and not same_distribution(base, actual):
                    raise ValueError('Ordinary distribution changed')
                freezes, layers = decision['boundary_freezes'], decision['executed_layers']
                if any(type(v) is not int for v in (freezes, layers)) or not 0 <= freezes <= layers <= 30:
                    raise ValueError('Invalid layer counters')
                pending = (ids, base, actual, freezes, layers)
            elif e['phase'] == 'committed':
                if pending is None or e['index'] != len(tokens):
                    raise ValueError('Invalid committed order')
                ids, base, actual, freezes, layers = pending
                token = e['token_id']
                value = metrics(base, actual, ids.index(token))
                group = groups[entropy_bin(value['base_entropy_bits'])]
                group['steps'] += 1
                group['kl_sum'] += value['kl_marked_to_base']
                group['tv_sum'] += value['total_variation']
                group['entropy_sum'] += value['base_entropy_bits']
                group['boundary_freezes'] += freezes
                group['executed_layers'] += layers
                group['steps_with_freezes'] += freezes > 0
                group['max_base_ge_99pct'] += 100 * max(base.weights) >= 99 * base.total
                group['eos_selected'] += token in eos_ids
                path_metrics.append(value)
                tokens.append(token)
                pending = None
    if pending is not None or tokens != row['committed_token_ids'] or not tokens:
        raise ValueError('Incomplete path')
    if row['completion'] != 'eos' or tokens[-1] not in eos_ids:
        raise ValueError('Complete EOS path required')
    return {'review_id': row['review_id'], 'condition': row['condition'],
            'bins': groups, 'steps': len(tokens),
            'sums': {k: math.fsum(v[k] for v in path_metrics) for k in path_metrics[0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'original', 'information', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if str(os.getppid()) != os.environ.get('KEYPRINT_WATCHDOG_PID'):
        raise RuntimeError('External watchdog required')
    checked = audit(args.study, args.original)
    if checked['audit_errors'] or checked['eos'] != 128:
        raise ValueError('Complete audited cohort required')
    read = lambda p: json.loads(p.read_text())
    information = read(args.information/'results.json')
    prior_plan = read(args.information/'plan.json')
    if (information['plan_sha256'] != digest(args.information/'plan.json')
            or prior_plan['cohort'] != checked):
        raise ValueError('Prior information result is not bound to this cohort')
    prior = {r['review_id']: r for r in information['rows']}
    rows = read(args.study/'private/runs.json')
    if len(prior) != 128 or set(prior) != {r['review_id'] for r in rows}:
        raise ValueError('Prior information cohort incomplete')
    # Pinned cohort EOS identity, verified in the committed prior profile audit.
    eos_ids = {248046}
    paths = [Path(__file__), Path(__file__).with_name('paced_information.py'),
             args.information/'results.json', args.information/'plan.json']
    bindings = {str(p): digest(p) for p in paths}
    plan = dict(schema='keyprint.paced-regimes-plan.v1', cohort=checked,
                bindings=bindings, entropy_cuts_bits=CUTS, eos_ids=sorted(eos_ids),
                selection='All 128 opened paths and all steps, including EOS',
                model_inference=False, scope=__doc__)
    args.output.mkdir(mode=0o700)
    write(args.output/'plan.json', plan)
    output = []
    with (args.output/'progress.jsonl').open('x') as f:
        for row in rows:
            result = inspect(args.study/'private'/row['review_id']/'journal.jsonl', row, eos_ids)
            if row['condition'] == 'marked':
                if (result['steps'] != prior[row['review_id']]['steps'] or any(
                        not math.isclose(v, prior[row['review_id']]['sums'][k], rel_tol=1e-12, abs_tol=1e-12)
                        for k, v in result['sums'].items())):
                    raise ValueError('Marked metrics differ from prior complete diagnostic')
            output.append(result)
            f.write(json.dumps(result)+'\n'); f.flush()
            print(json.dumps({'completed': len(output)}), flush=True)
    if audit(args.study, args.original) != checked or any(digest(Path(p)) != h for p,h in bindings.items()):
        raise ValueError('Inputs or helpers changed')
    groups = {}
    for arm in ('ordinary', 'marked'):
        chosen = [r for r in output if r['condition'] == arm]
        groups[arm] = {name: {k: math.fsum(r['bins'][name][k] for r in chosen)
                             for k in blank()} for name in LABELS}
    write(args.output/'results.json', dict(schema='keyprint.paced-regimes-results.v1',
          groups=groups, rows=output, plan_sha256=digest(args.output/'plan.json'),
          progress_sha256=digest(args.output/'progress.jsonl'),
          launch_ready=False, detector_calibrated=False, scope=__doc__))
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
