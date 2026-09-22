"""Read exported observations; use these same rows in a custom visualization."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    data = json.loads(args.recording.read_text())
    if data.get("schema") != "keyprint-comparison-v1":
        raise ValueError("Expected a Keyprint comparison export")
    result = data["experiment"]["outputs"]["marked"]
    print("Uncalibrated bit fractions, not detection confidence.")
    for point in result["inspection"]["series"]:
        print(json.dumps({**point, "text": result["text"][:point["characters"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
