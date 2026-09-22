"""Compare two local ByteLevel models sequentially; requires [transformers]."""
import argparse
from pathlib import Path
from keyprint import Keyprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    prompt = "In two sentences, explain how a seed becomes a tree."
    key = Keyprint.new_key()
    for index, path in enumerate((args.first, args.second), 1):
        with Keyprint.from_transformers(path, key=key) as wm:
            pair = wm.compare(prompt, max_tokens=192)
            print(pair.export(args.output / f"model-{index}"))


if __name__ == "__main__":
    main()
