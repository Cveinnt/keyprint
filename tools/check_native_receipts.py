"""Check retained final token paths against durable native-adapter selections."""
import argparse
import hashlib
import json
from pathlib import Path


def check(root: Path):
    traces = {}
    identities = []
    for path in sorted((root / "traces").glob("*.jsonl")):
        previous = "0" * 64
        tokens = []
        for sequence, line in enumerate(path.read_bytes().splitlines(keepends=True)):
            row = json.loads(line)
            if row["sequence"] != sequence or row["previous_sha256"] != previous:
                raise ValueError("journal chain mismatch: " + path.name)
            previous = hashlib.sha256(line).hexdigest()
            event = row["event"]
            if event["phase"] == "selected_tentative":
                tokens.append(event["token_id"])
            elif event["phase"] == "host_prefix_mismatch":
                raise ValueError("a host-prefix mismatch occurred: " + path.name)
            elif event["phase"] == "start":
                identities.append(event)
        traces[path.name] = tokens
    outputs = json.loads((root / "outputs.json").read_text())
    if not isinstance(outputs, list) or not outputs or len(traces) != len(outputs):
        raise ValueError("exactly one trace per returned request required")
    matched, tokens = set(), 0
    for output in outputs:
        ids = output.get("token_ids", output.get("output_ids"))
        if not isinstance(ids, list) or not ids:
            raise ValueError("actual returned token IDs required; text retokenization is insufficient")
        candidates = [name for name, selected in traces.items() if name not in matched and selected == ids]
        if len(candidates) != 1:
            raise ValueError("final host token path did not match one unique selection journal")
        matched.add(candidates[0])
        tokens += len(ids)
    return {"requests": len(outputs), "matching_tokens": tokens, "journal_chains": "pass",
            "scope": "final token paths match selected tokens; quality, detection and overhead unvalidated",
            "identities": identities}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(check(parser.parse_args().directory), indent=2))
