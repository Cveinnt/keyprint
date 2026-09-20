"""Profile optimized real MLX generation; instrumentation is not serving timing."""
import argparse
import cProfile
import importlib.metadata
import json
import os
from pathlib import Path
import pstats
import time

from benchmark_serving import require_storage, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path,
                        help="Profile every task in an existing confirmation plan at its frozen runtime")
    args = parser.parse_args()
    require_storage(args.output)
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    cases_path = (args.confirmation / 'public' / 'plan.json' if args.confirmation
                  else Path(__file__).with_name("inference_cases.json"))
    source = json.loads(cases_path.read_text())
    cases = source['cases'] if args.confirmation else source
    expected_runtime = source['expected_runtime_sha256'] if args.confirmation else None
    plan = {"script_sha256": sha(Path(__file__)), "cases_sha256": sha(cases_path),
            "cases": cases, "order": "Case order; ordinary first for even case index, marked first for odd",
            "scope": "One retained pair per fixed case under cProfile; diagnostic instrumentation, no serving acceptance",
            "failure_rule": "Retain all outcomes, no replacements or retries"}
    plan['expected_runtime_sha256'] = expected_runtime
    plan['source'] = 'existing_confirmation_plan' if args.confirmation else 'fixed_development_cases'
    (public / "plan.json").write_text(json.dumps(plan, indent=2))
    from keyprint import Keyprint
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(key)
    model = Keyprint.from_mlx(args.model, key=key, execution="experimental-fast")
    if expected_runtime is not None and model.identity['runtime_profile_sha256'] != expected_runtime:
        raise ValueError('Profiling runtime differs from the supplied confirmation plan')
    rows = []
    for index, case in enumerate(cases):
        order = ("ordinary", "marked") if index % 2 == 0 else ("marked", "ordinary")
        for condition in order:
            name = f"{case['id']}-{condition}"
            profiler = cProfile.Profile()
            row = {"id": name, "case": case["id"], "condition": condition}
            started = time.perf_counter()
            try:
                result = profiler.runcall(model.generate, case["prompt"], condition=condition,
                                          max_tokens=case["max_tokens"], output=args.output / name)
                row.update(text=result.text, usage=result.report["usage"],
                           completion=result.report["payload"]["completion"],
                           report_sha256=sha(args.output / name / "report.json"))
            except Exception as exc:
                row["error"] = type(exc).__name__
            row["instrumented_seconds"] = time.perf_counter() - started
            profiler.dump_stats(str(args.output / (name + ".prof")))
            stats = pstats.Stats(profiler)
            functions = [{"file": loc[0], "line": loc[1], "function": loc[2],
                          "primitive_calls": values[0], "calls": values[1],
                          "self_seconds": values[2], "cumulative_seconds": values[3]}
                         for loc, values in stats.stats.items()]
            row["functions"] = sorted(functions, key=lambda f: f["self_seconds"], reverse=True)
            (public / (name + ".json")).write_text(json.dumps(row, indent=2, ensure_ascii=False))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("text", "functions")}), flush=True)
    summary = {"status": "completed" if len(rows) == 2 * len(cases) and not any("error" in r for r in rows) else "incomplete",
               "outputs": len(rows), "errors": sum("error" in r for r in rows),
               "versions": {p: importlib.metadata.version(p) for p in ("keyprint", "mlx", "mlx-lm", "numpy")},
               "identity": model.identity, "scope": plan["scope"]}
    (public / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return int(summary["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
