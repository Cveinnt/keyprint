"""CLI routing, offline cache recovery, and rejection before model allocation."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from keyprint import Keyprint
from keyprint import cli


@pytest.fixture
def private_key(tmp_path):
    path = tmp_path / "key"
    path.write_bytes(bytes(range(32)))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("system,machine,expected", [
    ("Darwin", "arm64", "mlx"), ("Darwin", "x86_64", "transformers"),
    ("Linux", "aarch64", "transformers"), ("Windows", "AMD64", "transformers"),
])
@pytest.mark.parametrize("command", ["generate", "playground", "serve"])
def test_every_entry_routes_default_to_same_local_backend(monkeypatch, tmp_path, private_key,
                                                         system, machine, expected, command):
    # Platform simulation tests routing only, not inference support on those hosts.
    if command != "generate":
        pytest.importorskip("fastapi")
        uvicorn = pytest.importorskip("uvicorn")
        import keyprint.playground
        import keyprint.server
    calls = []
    result = SimpleNamespace(text="actual loader replaced by routing fixture", artifacts=tmp_path)
    candidate = SimpleNamespace(generate=lambda *a, **k: result)
    def load(backend):
        def fn(path, *, key):
            calls.append((backend, path, key))
            return candidate
        return fn
    monkeypatch.setattr(Keyprint, "from_mlx", staticmethod(load("mlx")))
    monkeypatch.setattr(Keyprint, "from_transformers", staticmethod(load("transformers")))
    monkeypatch.setattr(cli.platform, "system", lambda: system)
    monkeypatch.setattr(cli.platform, "machine", lambda: machine)
    def cached(backend):
        assert backend == expected
        return tmp_path
    monkeypatch.setattr(cli, "cached_model", cached)
    if command != "generate":
        monkeypatch.setattr(keyprint.playground, "create_playground", lambda loader, **kw: loader())
        monkeypatch.setattr(keyprint.server, "create_app", lambda loader, **kw: loader())
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    args = [command, "--key", str(private_key)]
    if command == "generate":
        args += ["--prompt", "Hello"]
    elif command == "serve":
        api_key = tmp_path / "api-key"
        api_key.write_bytes(bytes(reversed(range(32))))
        api_key.chmod(0o600)
        args += ["--api-key", str(api_key)]
    else:
        args += ["--output", str(tmp_path / "playground")]
    assert cli.main(args) == 0
    assert calls == [(expected, tmp_path, bytes(range(32)))]


def test_explicit_backend_and_model_override_cache_and_platform(monkeypatch, tmp_path, private_key):
    calls = []
    monkeypatch.setattr(cli, "default_backend", lambda: "mlx")
    monkeypatch.setattr(cli, "cached_model", lambda _: pytest.fail("explicit model must skip cache"))
    def load(path, *, key):
        calls.append(path)
        return SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(text="fixture", artifacts=tmp_path))
    monkeypatch.setattr(Keyprint, "from_transformers", staticmethod(load))
    assert cli.main(["generate", "--backend", "transformers", "--model", str(tmp_path),
                     "--key", str(private_key), "--prompt", "Hello"]) == 0
    assert calls == [tmp_path]


@pytest.mark.parametrize("extra", [
    ["--max-tokens", "0"], ["--max-tokens", "1025"], ["--max-tokens", "1.5"],
    ["--prompt", "   "], ["--prompt", "x" * 16001],
])
def test_invalid_generation_input_fails_before_key_or_model(monkeypatch, extra):
    monkeypatch.setattr(cli, "load_key", lambda _: pytest.fail("must reject before key access"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["generate", "--key", "missing", "--prompt", "hello", *extra])
    assert exc.value.code == 2


def test_bad_schema_rejected_before_model_load(monkeypatch, tmp_path, private_key, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"type":')
    monkeypatch.setattr(Keyprint, "from_mlx", staticmethod(lambda *a, **k: pytest.fail("model loaded")))
    assert cli.main(["generate", "--backend", "mlx", "--model", str(tmp_path), "--key", str(private_key),
                     "--prompt", "hello", "--json-schema", str(path)]) == 1
    assert "Keyprint:" in capsys.readouterr().err


@pytest.mark.parametrize("backend,model,revision", [
    ("mlx", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
    ("transformers", "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
])
def test_missing_cache_explains_exact_download_without_writing(monkeypatch, tmp_path, backend, model, revision):
    cache = tmp_path / "missing cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    with pytest.raises(ValueError) as exc:
        cli.cached_model(backend)
    message = str(exc.value)
    assert f"hf download {model} --revision {revision}" in message
    assert f"pip install '.[{backend}]'" in message
    assert "No download was started" in message
    assert not cache.exists()
    snapshot = cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert cli.cached_model(backend) == snapshot
    # Resolving a path does not qualify its assets: model loading still validates them.


def test_hf_home_respected_when_hub_override_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(ValueError, match=str(tmp_path / "hub")):
        cli.cached_model("transformers")
