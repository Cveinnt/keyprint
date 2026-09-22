"""Generate a real pair and export a remixable viewer. Requires the [mlx] extra."""
import argparse
from keyprint import Keyprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local pinned Qwen/MLX model directory")
    parser.add_argument("output", help="New directory for the recorded viewer")
    parser.add_argument("--prompt", default="In two sentences, explain how a seed becomes a tree.")
    args = parser.parse_args()
    with Keyprint.from_mlx(args.model, key=Keyprint.new_key()) as wm:
        pair = wm.compare(args.prompt, max_tokens=192)
        print(pair.marked)
        print(pair.export(args.output))


if __name__ == "__main__":
    main()
