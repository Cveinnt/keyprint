"""Summarize the registered concentration audit without selecting keys or steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(steps):
    if not steps:
        return {"steps": 0}
    return {"steps": len(steps),
            **{f"mean_{kind}_{metric}": statistics.mean(s[kind][metric] for s in steps)
               for kind in ('base', 'prepared')
               for metric in ('entropy_bits', 'max_probability', 'collision_probability')},
            "mean_total_variation": statistics.mean(s['total_variation'] for s in steps),
            "mean_kl_prepared_base_bits": statistics.mean(s['kl_prepared_base_bits'] for s in steps),
            "base_above_99_percent": sum(s['base']['max_probability'] > .99 for s in steps),
            "prepared_above_99_percent": sum(s['prepared']['max_probability'] > .99 for s in steps)}


def check_probability_receipt(row, payload):
    records = payload['sampling_records']
    steps = row['steps']
    if len(records) != len(steps) or payload['assigned_condition'] != row['condition']:
        raise ValueError('Observer and SDK receipt scope differ')
    if payload['committed_token_ids'] != [s['token_id'] for s in steps]:
        raise ValueError('Observer and SDK token IDs differ')
    for observed, recorded in zip(steps, records):
        if any(observed[left] != recorded[right] for left, right in (
            ('step', 'step'), ('token_id', 'token_id'),
            ('base_sha256', 'base_probability_sha256'),
            ('prepared_sha256', 'prepared_probability_sha256'))):
            raise ValueError('Observer probability hashes differ from SDK receipt')


def summarize(plan, rows):
    expected = {(k, c) for k in plan['keys'] for c in ('ordinary', 'marked')}
    actual = [(r['key_slot'], r['condition']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing, duplicate or undeclared attempts')
    summaries = []
    for row in rows:
        if row.get('error_type') or row.get('completion') != 'eos':
            raise ValueError('Failed or incomplete attempt; inspect retained results')
        steps = row['steps']
        if not steps or not all(s.get('reference_bitwise_equal') and s.get('layer_replay_bitwise_equal')
                                and 'token_id' in s for s in steps):
            raise ValueError('Missing probability parity or committed token evidence')
        if not row.get('exact_token_trace') or not row.get('model_forward_calls') or not row.get('draw_calls'):
            raise ValueError('Missing actual inference evidence')
        choice_steps = [s for s in steps if s['base']['max_probability'] <= .99]
        stages = {}
        for step in steps:
            for stage in step['stages']:
                stages.setdefault(stage['layer'], []).append(stage)
        summaries.append({"key_slot": row['key_slot'], "condition": row['condition'],
            "matches_prior_same_condition": row['matches_prior_same_condition'],
            "all_steps": aggregate(steps),
            "base_not_already_above_99_percent": aggregate(choice_steps),
            "layer_means": [{"layer": layer, "steps": len(values),
                             "entropy_bits": statistics.mean(v['entropy_bits'] for v in values),
                             "max_probability": statistics.mean(v['max_probability'] for v in values)}
                            for layer, values in sorted(stages.items())],
            "selected_path_log2_probability": {
                kind: math.fsum(math.log2(s[f'selected_{kind}_probability']) for s in steps)
                for kind in ('base', 'prepared')},
            "transformed_steps": sum(s['transformed'] for s in steps),
            "repeated_context_steps": sum(s['repeated_context'] for s in steps)})
    return {"attempts": len(rows), "committed_steps": sum(len(r['steps']) for r in rows),
            "unique_draw_transcripts": len({r['draw_transcript_sha256'] for r in rows}),
            "all_reference_probabilities_bitwise_equal": True,
            "rows": summaries,
            "limits": "One reused prompt and four existing keys. Same-prefix probabilities isolate the transform, not whole-answer quality. Intermediate layers use the 30-layer PRF domain. Observational timing cannot measure serving overhead. Path probability is conditional on recorded model heads; not a calibrated detector or general repeat rate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    public = args.root / 'public'
    plan = json.loads((public / 'plan.json').read_text())
    if digest(Path(__file__).with_name('audit_probability_concentration.py')) != plan['script_sha256']:
        raise ValueError('Registered audit script changed')
    if digest(Path(__file__).with_name('prose_quality_cases.json')) != plan['cases_sha256']:
        raise ValueError('Registered prompt changed')
    rows = json.loads((public / 'results.json').read_text())
    result = summarize(plan, rows)
    receipts = []
    for row in rows:
        path = args.root / 'private' / f"{row['key_slot']}-{row['condition']}" / 'report.json'
        report = json.loads(path.read_text())
        check_probability_receipt(row, report['report']['payload'])
        receipts.append({"key_slot": row['key_slot'], "condition": row['condition'],
                         "report_sha256": digest(path),
                         "runtime_profile_sha256": report['report']['target_identity']['runtime_profile_sha256'],
                         "steps": len(row['steps'])})
    result.update(all_observed_probability_hashes_match_sdk_receipts=True, receipts=receipts)
    result.update(plan_sha256=digest(public / 'plan.json'), results_sha256=digest(public / 'results.json'),
                  summary_script_sha256=digest(Path(__file__)))
    (public / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('attempts', 'committed_steps', 'unique_draw_transcripts')}))


if __name__ == '__main__':
    main()
