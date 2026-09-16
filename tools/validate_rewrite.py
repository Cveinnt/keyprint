"""Small paired fidelity screen with retained failures; not quality acceptance."""
import argparse
import json
import os
from pathlib import Path
import time

from keyprint import Keyprint, KeyprintError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    key = Keyprint.new_key()
    fd = os.open(args.output / "owner.key", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(key)
    candidate = Keyprint.from_mlx(args.model, key=key)
    fixtures = json.loads(Path(__file__).with_name("rewrite_fixtures.json").read_text())
    runs = []
    for index, fixture in enumerate(fixtures):
        order = ("ordinary", "marked") if index % 2 == 0 else ("marked", "ordinary")
        for condition in order:
            started = time.monotonic()
            try:
                result = candidate.rewrite(fixture["text"], max_tokens=384, condition=condition,
                                          output=args.output / (fixture["id"] + "-" + condition))
                row = {**fixture, "condition": condition, "candidate": result.text,
                       "status": result.status, "checks": result.checks, "usage": result.generation.report["usage"]}
            except KeyprintError as exc:
                row = {**fixture, "condition": condition, "error": exc.report}
            row["seconds"] = time.monotonic() - started
            runs.append(row)
            (args.output / "summary.json").write_text(json.dumps({"scope": "12-run paired lexical fidelity screen; no semantic or detector acceptance", "runs": runs}, ensure_ascii=False, indent=2))
            print(fixture["id"], condition, row.get("checks", "error"), flush=True)


if __name__ == "__main__":
    main()
