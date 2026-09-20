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
@pytest.mark.parametrize("execution", ["reference", "experimental-fast", "experimental-native"])
def test_every_entry_routes_default_to_same_local_backend(monkeypatch, tmp_path, private_key,
                                                         system, machine, expected, command, execution):
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
        def fn(path, *, key, **options):
            calls.append((backend, path, key, options))
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
    if execution != "reference":
        args += ["--execution", execution]
    if expected != "mlx" and execution != "reference":
        monkeypatch.setattr(cli, "load_key", lambda _: pytest.fail("must reject before key access"))
        assert cli.main(args) == 1
        assert calls == []
        assert not (tmp_path / "playground").exists()
        return
    assert cli.main(args) == 0
    assert calls == [(expected, tmp_path, bytes(range(32)),
                      {"execution": execution} if expected == "mlx" else {})]


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


@pytest.fixture
def preflight(monkeypatch, tmp_path):
    """Only dependency discovery is simulated; filesystem failures stay real."""
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(metadata=cli.importlib.metadata,
                                                        import_module=lambda _: SimpleNamespace()))
    monkeypatch.setattr(Keyprint, "from_mlx", staticmethod(lambda *a, **k: pytest.fail("model loaded")))
    monkeypatch.setattr(Keyprint, "from_transformers", staticmethod(lambda *a, **k: pytest.fail("model loaded")))
    model = tmp_path / "model with spaces"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(b"presence only, not valid weights")
    return model


def test_doctor_reports_limited_transformers_scope_and_quoted_command(preflight):
    import shlex
    result = cli.doctor(playground=True, backend="transformers", model=preflight)
    assert result["status"] == "pass"
    assert "weights and inference not verified" in result["model_check"]
    assert shlex.split(result["next_command"]) == ["keyprint", "playground", "--backend",
        "transformers", "--model", str(preflight), "--execution", "reference"]


@pytest.mark.parametrize("failure", ["dependency", "metadata", "weights", "missing directory"])
def test_doctor_reports_actionable_preflight_failures(preflight, monkeypatch, failure):
    if failure == "dependency":
        def missing(name):
            if name == "fastapi":
                raise ImportError("missing test dependency")
        monkeypatch.setattr(cli.importlib, "import_module", missing)
        expected = "Cannot import fastapi"
    elif failure == "metadata":
        (preflight / "config.json").write_text("{")
        expected = "Expecting property name"
    elif failure == "weights":
        (preflight / "model.safetensors").unlink()
        expected = "weights are missing"
    else:
        preflight = preflight / "absent"
        expected = "existing local directory"
    result = cli.doctor(playground=True, backend="transformers", model=preflight)
    assert result["status"] == "fail"
    assert any(expected in problem for problem in result["problems"])


def test_doctor_checks_pinned_mlx_assets_and_native_version_without_loading(preflight, monkeypatch):
    from keyprint.backends import mlx
    from keyprint.experimental import native_mlx
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native_mlx, "native_backend", lambda: SimpleNamespace(identity={"package_version": "0.1.0a2"}))
    paths = []
    monkeypatch.setattr(mlx, "verify_assets", lambda path: paths.append(path))
    result = cli.doctor(playground=True, backend="mlx", model=preflight, execution="experimental-native")
    assert result["status"] == "pass"
    assert result["native_accelerator"] == "0.1.0a2"
    assert paths == [preflight]
    def corrupted(path):
        raise ValueError("pinned asset mismatch")
    monkeypatch.setattr(mlx, "verify_assets", corrupted)
    assert "pinned asset mismatch" in cli.doctor(playground=True, backend="mlx", model=preflight)["problems"]
    def missing():
        raise ImportError("matching native wheel required")
    monkeypatch.setattr(native_mlx, "native_backend", missing)
    paths.clear()
    assert "matching native wheel required" in cli.doctor(playground=True, backend="mlx", model=preflight,
                                                          execution="experimental-native")["problems"]
    assert not paths


def test_doctor_rejects_mlx_on_other_hosts_before_imports(preflight, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.importlib, "import_module", lambda _: pytest.fail("backend imported"))
    result = cli.doctor(playground=True, backend="mlx", model=preflight)
    assert result["status"] == "fail"
    assert "MLX requires Apple Silicon" in result["problems"][0]


def test_plain_doctor_does_not_discover_models_or_import_optional_dependencies(monkeypatch):
    monkeypatch.setattr(cli, "cached_model", lambda _: pytest.fail("cache accessed"))
    monkeypatch.setattr(cli.importlib, "import_module", lambda _: pytest.fail("optional import"))
    assert cli.doctor()["status"] == "pass"


def test_cli_doctor_json_and_exit_status(preflight, capsys):
    import json
    args = ["doctor", "--playground", "--backend", "transformers", "--model", str(preflight), "--json"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    (preflight / "model.safetensors").unlink()
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
    assert cli.main(["doctor", "--backend", "mlx"]) == 1
    assert "doctor --playground" in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["mlx", "transformers"])
def test_download_fetches_only_pinned_files_in_selected_cache(monkeypatch, tmp_path, backend, capsys):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    imports, calls = [], []
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda name: imports.append(name)))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")
    monkeypatch.setattr(hub, "snapshot_download", download)
    assert cli.download_model(backend) == tmp_path / "snapshot"
    model, revision, files, _ = cli.pinned_model(backend)
    assert calls == [{"repo_id": model, "revision": revision, "cache_dir": tmp_path / "cache",
                      "allow_patterns": files, "token": False}]
    assert len(revision) == 40 and not any("*" in name for name in files)
    assert "model.safetensors" in files and not any(name.endswith((".py", ".bin")) for name in files)
    assert imports == (["mlx_lm"] if backend == "mlx" else ["torch", "transformers"])
    assert "Existing cached files are reused" in capsys.readouterr().out


def test_download_failure_preserves_partial_cache_without_retry(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda _: None))
    calls = []
    partial = tmp_path / "partial"
    def interrupted(**kwargs):
        calls.append(kwargs)
        partial.write_text("retained")
        raise OSError("connection interrupted")
    monkeypatch.setattr(hub, "snapshot_download", interrupted)
    with pytest.raises(RuntimeError, match="Retry --download explicitly"):
        cli.download_model("transformers")
    assert len(calls) == 1 and partial.read_text() == "retained"


def test_missing_backend_rejected_before_download(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    def missing(_):
        raise ImportError("missing torch")
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=missing))
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: pytest.fail("download started"))
    with pytest.raises(ImportError, match="Install backend dependencies first"):
        cli.download_model("transformers")


def test_mlx_download_rejected_on_unsupported_host_before_imports(monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda _: pytest.fail("imported")))
    with pytest.raises(ValueError, match="Apple Silicon"):
        cli.download_model("mlx")


@pytest.mark.parametrize("extra", [["--model", "existing"], ["--port", "80"],
    ["--backend", "transformers", "--execution", "experimental-native"]])
def test_invalid_download_request_fails_before_network_or_output(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(cli, "download_model", lambda _: pytest.fail("download started"))
    output = tmp_path / "private"
    assert cli.main(["playground", "--download", "--output", str(output), *extra]) == 1
    assert not output.exists()


@pytest.mark.parametrize("download", [False, True])
def test_playground_download_is_explicit_and_uses_returned_path(monkeypatch, tmp_path, download):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    import keyprint.playground
    calls = []
    def cached(_):
        assert not download
        return tmp_path
    def fetch(_):
        assert download
        calls.append("download")
        return tmp_path
    monkeypatch.setattr(cli, "cached_model", cached)
    monkeypatch.setattr(cli, "download_model", fetch)
    monkeypatch.setattr(Keyprint, "from_transformers", staticmethod(lambda path, **kw: calls.append(path)))
    monkeypatch.setattr(keyprint.playground, "create_playground", lambda loader, **kw: loader())
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    # Supply a key so the mocked server need not create its private directory.
    key = tmp_path / "key"
    key.write_bytes(bytes(32)); key.chmod(0o600)
    args = ["playground", "--backend", "transformers", "--key", str(key), "--output", str(tmp_path / "output")]
    assert cli.main(args + (["--download"] if download else [])) == 0
    assert calls == (["download", tmp_path] if download else [tmp_path])


def test_invalid_key_rejected_before_download(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    monkeypatch.setattr(cli, "download_model", lambda _: pytest.fail("download started"))
    assert cli.main(["playground", "--download", "--key", str(tmp_path / "missing")]) == 1


def test_missing_native_wheel_rejected_before_download(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from keyprint.experimental import native_mlx
    def missing():
        raise ImportError("matching native wheel required")
    monkeypatch.setattr(native_mlx, "native_backend", missing)
    monkeypatch.setattr(cli, "download_model", lambda _: pytest.fail("download started"))
    assert cli.main(["playground", "--backend", "mlx", "--execution", "experimental-native", "--download"]) == 1
