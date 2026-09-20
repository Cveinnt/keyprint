"""Reconcile the three-path engine comparison and display every retained output."""
import argparse
import hashlib
import html
import inspect
import json
from pathlib import Path

from analyze_mlx_capacity import read_journal
from audit_capped_rendering import verify_rendering
from audit_serving import verify_output
from benchmark_engine_baseline import ARMS, REPEATS, analyze
from benchmark_serving import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root, public = args.run, args.run / "public"
    destination = public / "integrity.json"
    if destination.exists(): raise ValueError("Never overwrite an existing audit")
    plan = json.loads((public / "plan.json").read_text())
    load = json.loads((public / "load.json").read_text())
    summary = json.loads((public / "summary.json").read_text())
    for filename, field in [("benchmark_engine_baseline.py", "script_sha256"),
                            ("benchmark_serving.py", "analysis_sha256"),
                            ("inference_cases.json", "cases_sha256")]:
        if sha(Path(__file__).with_name(filename)) != plan[field]:
            raise ValueError("Frozen benchmark, analysis or prompts differ")
    cases = json.loads(Path(__file__).with_name("inference_cases.json").read_text())
    if plan["cases"] != cases or plan["arms"] != list(ARMS) or plan["repeats"] != REPEATS:
        raise ValueError("Declared workload differs")
    from keyprint.experimental.fast_reporting import _experimental_target
    from keyprint._engine.legacy._impl.research.token_runtime_binding import runtime_binding
    from keyprint._engine.legacy._impl.research.token_channel_host import EOS
    from keyprint.backends.mlx import ASSETS
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import AutoTokenizer
    _experimental_target(load["identity"])
    sources = {Path(inspect.getfile(f)).name: sha(Path(inspect.getfile(f)))
               for f in (stream_generate, make_sampler, TokenizerWrapper)}
    if sources != plan["engine_source_sha256"]: raise ValueError("Upstream engine source differs")
    model = Path(load["model_path"])
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha(model / name) != ASSETS[name]: raise ValueError("Tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    wrapper = TokenizerWrapper(tokenizer)
    token_bytes = runtime_binding(max_steps=load["identity"]["max_steps"]).token_bytes
    expected = [(cases[0], -1, arm, True) for arm in ARMS]
    expected += [(case, repeat, arm, False) for repeat in range(REPEATS) for case in cases for arm in ARMS]
    rows, warmups, byte_checks, tokens, engine_tokens = [], [], [], 0, 0
    for case, repeat, arm, warmup in expected:
        name = f"{'warmup-' if warmup else ''}{case['id']}-{repeat}-{arm}"
        row = json.loads((public / (name + ".json")).read_text())
        cap = 32 if warmup else case["max_tokens"]
        if (row["id"] != name or row["case"] != case["id"] or row["repeat"] != repeat
                or row["arm"] != arm or row["max_tokens"] != cap or "error" in row):
            raise ValueError("Failed, missing or substituted output")
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids = row["token_ids"]
        if (len(ids) != row["completion_tokens"] or not 1 <= len(ids) <= cap
                or any(type(i) is not int or i < 0 for i in ids)
                or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]):
            raise ValueError("Token count or output hash differs")
        if arm == "engine":
            if row["prompt_ids"] != prompt:
                raise ValueError("Upstream prompt differs")
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            if decoded != row["decoded_text"] or row["native_rendering_matches_decode"] != (decoded == row["text"]):
                raise ValueError("Upstream decoding metadata differs")
            is_eos = ids[-1] in wrapper.eos_token_ids
            if ((row["completion"] == "eos") != is_eos
                    or any(i in wrapper.eos_token_ids for i in ids[:-1])
                    or row["completion"] not in ("eos", "length")
                    or (row["completion"] == "length" and len(ids) != cap)):
                raise ValueError("Upstream finish reason differs from token path")
            # Reconstruct the unmodified upstream rendering, including cap behavior.
            detokenizer = wrapper.detokenizer
            segments = []
            for index, token in enumerate(ids):
                if token in wrapper.eos_token_ids: break
                detokenizer.add_token(token)
                if index + 1 != cap: segments.append(detokenizer.last_segment)
            detokenizer.finalize()
            segments.append(detokenizer.last_segment)
            if "".join(segments) != row["text"]:
                raise ValueError("Upstream renderer replay differs")
            engine_tokens += len(ids)
        else:
            folder = root / name
            for filename, field in (("report.json", "report_sha256"), ("journal.jsonl", "journal_sha256")):
                if sha(folder / filename) != row[field]: raise ValueError("SDK artifact changed")
            report = json.loads((folder / "report.json").read_text())["report"]
            events = read_journal(folder / "journal.jsonl")
            verify_output(row, report, events, prompt, load["identity"])
            if ids != report["payload"]["committed_token_ids"]: raise ValueError("Public SDK token path differs")
            byte_checks.append({"id": name, **verify_rendering(report, events, token_bytes, EOS)})
        tokens += len(ids)
        (warmups if warmup else rows).append(row)
    analysis = analyze(rows, [c["id"] for c in cases])
    if (summary["status"] != "completed" or summary["requests"] != 72 or summary["warmups"] != 3
            or summary["failure"] is not None or summary["analysis"] != analysis
            or summary["production_overhead_accepted"] is not False):
        raise ValueError("Summary differs or overclaims acceptance")
    sections = []
    names = {"engine": "MLX-LM", "sdk_ordinary": "Keyprint ordinary", "sdk_marked": "Keyprint marked"}
    for repeat in range(REPEATS):
        for case in cases:
            columns = []
            for arm in ARMS:
                row = next(r for r in rows if (r["repeat"], r["case"], r["arm"]) == (repeat, case["id"], arm))
                columns.append(f'<article><h3>{names[arm]}</h3><p>{row["completion_tokens"]} tokens · {row["seconds"]:.3f}s · {row["completion"]}</p><pre>{html.escape(row["text"])}</pre></article>')
            sections.append(f'<section><h2>{html.escape(case["id"])} · repeat {repeat + 1}</h2><p>{html.escape(case["prompt"])}</p><div class="triple">'+"".join(columns)+"</div></section>")
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Keyprint engine comparison</title>
<style>body{max-width:1400px;margin:48px auto;padding:0 24px;background:#f7f5ee;color:#292923;font:18px/1.55 Georgia,serif}h1{font-size:42px}section{border-top:1px solid #ccc6b7;padding:24px 0}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}article{background:#fffdf8;padding:20px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 Georgia,serif}@media(max-width:850px){.triple{grid-template-columns:1fr}}</style>
<h1>One model. Three complete paths.</h1><p>All 72 measured outputs, including caps. Native sampling and bookkeeping differ. Timing does not approve meaning, detection or production serving. Three declared warmups remain in the JSON receipts.</p>'''
    (public / "comparison.html").write_text(page + "".join(sections))
    result = {"status": "pass", "measured_outputs": len(rows), "warmups": len(warmups),
              "verified_tokens": tokens, "engine_tokens": engine_tokens,
              "sdk_byte_checks": byte_checks, "analysis": analysis,
              "plan_sha256": sha(public / "plan.json"), "summary_sha256": sha(public / "summary.json"),
              "auditor_sha256": sha(Path(__file__)),
              "scope": "Stored artifacts, token counts, native renderer replay, SDK byte/journal audit and arithmetic; not model-head replay or semantic acceptance"}
    with destination.open("x") as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "sdk_byte_checks"}))


if __name__ == "__main__": main()
