import hashlib

import pytest

from keyprint.experimental import sglang_runtime


@pytest.fixture
def source(monkeypatch, tmp_path):
    path = tmp_path / "srt/sampling/custom_logit_processor.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"verified callback source\n")
    monkeypatch.setattr(sglang_runtime, "SOURCE_FILES", {
        "srt/sampling/custom_logit_processor.py": hashlib.sha256(path.read_bytes()).hexdigest()
    })
    return tmp_path, path


@pytest.mark.parametrize("version", sorted(sglang_runtime.VERSIONS))
def test_known_metadata_labels_require_identical_source(source, version):
    root, path = source
    result = sglang_runtime.verify_runtime(version, root)
    assert result["build_version"] == version
    assert result["source_revision"] == sglang_runtime.REVISION
    path.write_bytes(b"different callback implementation\n")
    with pytest.raises(ValueError, match="source differs"):
        sglang_runtime.verify_runtime(version, root)


def test_missing_source_and_unknown_version_fail_closed(source):
    root, path = source
    with pytest.raises(ValueError, match="not qualified"):
        sglang_runtime.verify_runtime("0.5.22", root)
    path.unlink()
    with pytest.raises(ValueError, match="source is unavailable"):
        sglang_runtime.verify_runtime(next(iter(sglang_runtime.VERSIONS)), root)
