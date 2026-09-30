"""Model-free repository checks. Standard library only; no external requests."""
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

from serve_demos import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    docs = ["README.md", "USAGE.md", "CONTRIBUTING.md", "COMMUNITY.md", "AGENTS.md",
            "ROADMAP.md", "docs/README.md", "docs/maintainers/CI.md",
            "demos/README.md", "examples/README.md", "llms.txt"]
    for name in docs:
        source = ROOT / name
        for target in re.findall(r'\]\(([^\s)]+)\)', source.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            path = source.parent / target.split("#", 1)[0]
            if not path.exists():
                raise ValueError(f"Broken local link: {name} -> {target}")
    python_files = sorted((ROOT / "src/keyprint").rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    recordings = ROOT / "demos/recordings"
    provenance = json.loads((recordings / "provenance.json").read_text())
    for name, field in [("gallery.json", "gallery_sha256"), ("replay.json", "recording_sha256")]:
        digest = hashlib.sha256((recordings / name).read_bytes()).hexdigest()
        if digest != provenance[field]:
            raise ValueError(f"Public recording differs from retained provenance: {name}")
    gallery = json.loads((recordings / "gallery.json").read_text())
    for example in gallery["examples"]:
        for condition, output in example["recording"]["experiment"]["outputs"].items():
            if "".join(token["text"] for token in output["trace"]) != output["text"]:
                raise ValueError(f"Trace/text mismatch: {example['title']} / {condition}")
    with tempfile.TemporaryDirectory(prefix="keyprint-community-check-") as temporary:
        destination = Path(temporary) / "play"
        build(destination)
        with ZipFile(destination / "keyprint-demo.zip") as archive:
            for name in ("gallery.json", "replay.json", "provenance.json"):
                if archive.read(name) != (recordings / name).read_bytes():
                    raise ValueError(f"Demo export changed {name}")
            if b'data-mode="replay"' not in archive.read("index.html"):
                raise ValueError("Public gallery must use recorded mode")
    print(f"Checked {len(docs)} entry docs, {len(python_files)} Python source files, "
          f"{len(gallery['examples'])} original pairs and the downloadable demo.")
    print("No model inference or SDK compatibility qualification was performed.")


if __name__ == "__main__":
    main()
