"""Integrity of the namespaced derivative; not an authenticity signature."""
from pathlib import Path
import hashlib
import json


def verify() -> dict[str, object]:
    root = Path(__file__).parent / "_engine"
    manifest = root / "port-manifest.json"
    data = json.loads(manifest.read_text())
    for name, info in data["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != info["ported_sha256"]:
            raise RuntimeError(f"Keyprint source integrity mismatch: {name}")
    return {"status": "pass", "files": len(data["files"]),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "scope": "Namespaced derivative integrity; no detector or scientific acceptance verdict."}
