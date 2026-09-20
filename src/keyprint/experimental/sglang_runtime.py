"""Identify the pinned CPU callback contract independently of Git tag metadata."""
import hashlib
from pathlib import Path

REVISION = "13d593b6cf885c5c4d50eea88c82b9e28cf5941e"
# Both were produced by the same pinned checkout as upstream tags evolved.
VERSIONS = frozenset({"0.5.20.dev791+g13d593b6c", "0.5.21.dev69+g13d593b6c"})
SOURCE_FILES = {
    "srt/managers/schedule_batch.py": "179666548b89f151398d45ae34a9b676f5a1b42939d180d2c8051cd990807392",
    "srt/sampling/custom_logit_processor.py": "ec50329f4488d3fa0f7f521850b706170ac3609836d08c54a7e13bc2731f5bbd",
    "srt/sampling/sampling_params.py": "7295f275a1a286ae5a4414bcedd1d48a3a9824a0c2a5c915e091ed06b6f74730",
    "srt/layers/sampler.py": "78fa19f0e23a3d49f3aed07cd932e7a252c04d009750305cc456ba0f1e961390",
    "srt/server_args.py": "118ebd9aa79757110c52ddd08593f05972c468f4326dc142c33e49a8b2702c65",
}


def verify_runtime(version: str, package_root: Path) -> dict[str, object]:
    """Check the known build label and source files used by this adapter.

    This fingerprints the Python sampling contract, not the compiled kernels or
    every upstream file. The pinned Docker build and actual inference provide
    separate runtime evidence. Changed files and unknown versions fail closed.
    """
    if version not in VERSIONS:
        raise ValueError(f"SGLang CPU build {version!r} is not qualified; use source revision {REVISION}")
    for relative, expected in SOURCE_FILES.items():
        path = package_root / relative
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"SGLang runtime source is unavailable: {relative}") from exc
        if actual != expected:
            raise ValueError(f"SGLang runtime source differs: {relative}; use source revision {REVISION}")
    return {"source_revision": REVISION, "build_version": version,
            "source_files_sha256": dict(SOURCE_FILES),
            "scope": "Pinned Python sampling contract; not compiled-kernel or production qualification"}
