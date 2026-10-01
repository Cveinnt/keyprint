import hashlib
import pytest
from keyprint.backends import mlx


@pytest.fixture
def assets(tmp_path, monkeypatch):
    hashes = {}
    for name in mlx.ASSETS:
        data = ("fixture:" + name).encode()
        (tmp_path / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(mlx, "ASSETS", hashes)
    return tmp_path


def test_appledouble_metadata_is_ignored_but_retained(assets):
    sidecar = assets / "._model.safetensors"
    metadata = bytes.fromhex("0005160700020000") + bytes(4088)
    sidecar.write_bytes(metadata)
    mlx.verify_assets(assets)
    assert sidecar.read_bytes() == metadata
    assert [p.name for p in assets.glob("model*.safetensors")] == ["model.safetensors"]


@pytest.mark.parametrize("data", [b"", b"not metadata", bytes.fromhex("0005160700000001")])
def test_unknown_sidecar_is_still_rejected(assets, data):
    (assets / "._model.safetensors").write_bytes(data)
    with pytest.raises(ValueError, match="expected exactly"):
        mlx.verify_assets(assets)


def test_extra_real_weights_are_still_rejected(assets):
    (assets / "model-extra.safetensors").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="expected exactly"):
        mlx.verify_assets(assets)


def test_metadata_does_not_bypass_weight_hash(assets):
    (assets / "._model.safetensors").write_bytes(bytes.fromhex("0005160700020000"))
    (assets / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="model asset mismatch"):
        mlx.verify_assets(assets)


def test_metadata_is_not_a_substitute_for_weights(assets):
    (assets / "model.safetensors").unlink()
    (assets / "._model.safetensors").write_bytes(bytes.fromhex("0005160700020000"))
    with pytest.raises(ValueError, match="expected exactly"):
        mlx.verify_assets(assets)
