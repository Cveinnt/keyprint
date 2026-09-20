"""Compare complete durable callers on independently executed real inference.

Fixture randomness is passed explicitly to each caller. No production RNG or
reference module is patched. Results qualify parity only, not serving speed,
semantic quality or detection. Every failure and truncated output is retained.
"""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import random

from benchmark_serving import require_storage, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=["reference", "experimental-fast", "experimental-native"], default="experimental-fast")
    args = parser.parse_args()
    require_storage(args.output)
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    from keyprint import Keyprint
    from keyprint.experimental.fast_public import FastPublicCandidate
    from keyprint.backends.mlx_bounded import BoundedReferencePublicCandidate
    from keyprint._engine.research.keyprint_candidate_v3_caller import DurableJournal
    from keyprint.backends.mlx import MLXModel
    cases_path = Path(__file__).with_name("inference_cases.json")
    cases = json.loads(cases_path.read_text())
    reference = Keyprint(key=bytes(range(32)))._candidate
    label = "bounded_reference" if args.execution == "reference" else "native" if args.execution == "experimental-native" else "fast"
    if args.execution == "experimental-native":
        from keyprint.experimental.native_mlx import NativePublicCandidate
        tested = NativePublicCandidate(reference)
    else:
        tested = BoundedReferencePublicCandidate(reference) if args.execution == "reference" else FastPublicCandidate(reference)
    plan = {
        "scope": __doc__, "cases": cases, "script_sha256": sha(Path(__file__)),
        "cases_sha256": sha(cases_path), "reference_identity": reference.core_identity,
        label + "_identity": tested.core_identity, "candidate_label": label,
        "execution": args.execution, "model_path": str(args.model.resolve()),
        "versions": {p: importlib.metadata.version(p) for p in ("keyprint", "mlx", "mlx-lm", "numpy")},
        "key": "Public fixture bytes 0..31",
        "randomness": "Independent Random(20260920 + case index) per execution and condition",
        "order": "Alternate frozen reference/tested execution first by case index; fresh caller cache for each execution",
        "failure_rule": "Retain all attempts without retries or replacements",
    }
    compact = args.execution == 'experimental-native' and tested.core_identity['version'].endswith('-v2')
    if compact:
        from commitment_parity import observe_native_vectors, legacy_sampling_projection, validate_observed_sequence
        plan['vector_observation'] = {
            'scope': 'Test-only native vector capture; reference unchanged; no serving timing',
            'sources': {name: sha(Path(__file__).with_name(name))
                        for name in ('commitment_parity.py', 'vector_commitment.py')},
        }
    (public / "plan.json").write_text(json.dumps(plan, indent=2))
    backend = MLXModel.load(args.model)
    rows = []
    for index, case in enumerate(cases):
        prompt = backend.encode_prompt(case["prompt"])
        for condition in ("ordinary", "marked"):
            row = {"case": case["id"], "condition": condition, "max_tokens": case["max_tokens"],
                   "executions": {}, "checks": {}, "errors": []}
            reports = {}
            observations = []
            order = (("reference", reference), (label, tested))
            if index % 2: order = tuple(reversed(order))
            for name, candidate in order:
                directory = args.output / f"{case['id']}-{condition}-{name}"
                directory.mkdir()
                reservations = {}
                def reserve(action, metadata):
                    reservations[action] = reservations.get(action, 0) + 1
                try:
                    with (observe_native_vectors() if compact and name == label else nullcontext([])) as captured, DurableJournal(directory / "journal.jsonl") as journal:
                        report = candidate.run_response(
                            backend.model, prompt, key=bytes(range(32)), condition=condition,
                            random_bits=random.Random(20260920 + index).getrandbits,
                            journal=journal, reserve=reserve, max_tokens=case["max_tokens"],
                            max_model_calls=case["max_tokens"] + 8,
                            allow_thinking=False, allow_tools=False)
                    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
                    reports[name] = report
                    if compact and name == label:
                        observations = captured
                        (directory / 'vectors.json').write_text(json.dumps(observations))
                    row["executions"][name] = {
                        "kind": report["kind"], "reservations": reservations,
                        "report_sha256": sha(directory / "report.json"),
                        "journal_sha256": sha(directory / "journal.jsonl"),
                        "text": (report.get("rendered_carriers") or {}).get("visible_text"),
                    }
                    if compact and name == label:
                        row['executions'][name]['vectors_sha256'] = sha(directory / 'vectors.json')
                except Exception as exc:
                    row["errors"].append({"execution": name, "type": type(exc).__name__, "message": str(exc)})
            if len(reports) == 2 and all(r["kind"] == "generation_trace" for r in reports.values()):
                a, b = reports["reference"], reports[label]
                for field in ("committed_token_ids", "sampling_records", "completion", "literal_diagnostics"):
                    value = b['payload'][field]
                    if compact and field == 'sampling_records':
                        validate_observed_sequence(observations, len(value))
                        value = legacy_sampling_projection(value, observations)
                    row["checks"][field] = a["payload"][field] == value
                row["checks"]["rendered_text"] = a["rendered_carriers"] == b["rendered_carriers"]
                row["checks"]["distinct_execution_identity"] = a["target_identity"] != b["target_identity"]
                row["checks"]["same_consumed_work"] = row["executions"]["reference"]["reservations"] == row["executions"][label]["reservations"]
                row["tokens_per_execution"] = len(a["payload"]["committed_token_ids"])
                row["completion"] = a["payload"]["completion"]
            else:
                row["checks"]["both_generations_completed"] = False
            row["status"] = "pass" if row["checks"] and all(row["checks"].values()) and not row["errors"] else "failed"
            (public / f"{case['id']}-{condition}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "executions"}), flush=True)
    summary = {"status": "pass" if len(rows) == 2 * len(cases) and all(r["status"] == "pass" for r in rows) else "failed",
               "pairs": len(rows), "outputs": sum(len(r["executions"]) for r in rows),
               "tokens_per_execution": sum(r.get("tokens_per_execution", 0) for r in rows),
               "scope": "Complete caller parity on independently executed model paths; no performance or quality acceptance"}
    (public / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return int(summary["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
