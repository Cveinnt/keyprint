"""Generate three real pairs and export a static gallery. Requires [mlx]."""
import argparse
from keyprint import Keyprint, export_gallery

PROMPTS = {
    "A small explanation": "In two sentences, explain how a seed becomes a tree to a curious adult.",
    "An everyday email": "Write a warm, brief email inviting a friend to a picnic this weekend. Do not invent a date, time, location, or names. Use at most 60 words.",
    "Une idée en français": "Explique en français pourquoi le ciel est bleu. Utilise deux phrases simples, sans traduire ta réponse en anglais.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local pinned Qwen/MLX model directory")
    parser.add_argument("output", help="New directory for the recorded gallery")
    args = parser.parse_args()
    with Keyprint.from_mlx(args.model, key=Keyprint.new_key()) as wm:
        pairs = {title: wm.compare(prompt, max_tokens=192) for title, prompt in PROMPTS.items()}
        print(export_gallery(pairs, args.output))


if __name__ == "__main__":
    main()
