"""Installed-wheel regressions, including comparisons to the preserved release."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event

import numpy as np
import pytest

from keyprint import Keyprint, KeyprintError, KeyprintCancelled, verify

KEY = bytes(range(32))
ROOT = Path(__file__).resolve().parents[1]


def head(*tokens):
    result = np.full((1, 151936), -np.inf, dtype=np.float32)
    result[0, list(tokens)] = 0
    return result


class FakeBackend:
    array = staticmethod(np.array)
    int32 = np.int32
    float32 = np.float32
    eval = staticmethod(lambda value: None)


def test_import_does_not_pollute_process(tmp_path):
    result = subprocess.run([sys.executable, "-c", '''
import sys
before = sys.path[:]
from keyprint import Keyprint
k = Keyprint(key=bytes(range(32)))
assert sys.path == before
assert not any(n.startswith('keyprint_candidate') for n in sys.modules)
assert 'torch' not in sys.modules and 'mlx' not in sys.modules
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key", [b"short", "a" * 32, bytearray(32), bytes(33)])
def test_invalid_key_rejected(key):
    with pytest.raises(ValueError):
        Keyprint(key=key)


@pytest.mark.skipif(not (ROOT / "sdk/keyprint_v3/__init__.py").exists(),
                    reason="reference parity needs the repository's preserved SDK tree")
def test_reference_parity(tmp_path):
    # Separate interpreters prevent importing the old nested keyprint package
    # over the new public namespace. Compare behavior, not re-labeled hashes.
    script = '''
import json, random, numpy as np
{setup}
results = []
for condition in ('ordinary', 'marked'):
  for seed in range(8):
    draws = random.Random(seed)
    with {pipeline} as p:
      for step in range(16):
        h = np.full((1,151936), -np.inf, dtype=np.float32)
        h[0,32:40] = np.linspace(-step/4,1,8,dtype=np.float32)
        r = p.step(h, draws.getrandbits)
        assert r['kind'] != 'error'
      r = p.finish()
    results.append([r['kind'], r['payload']['committed_token_ids'], r['rendered_carriers']])
print(json.dumps(results,sort_keys=True))
'''
    old = script.format(setup="from keyprint_v3 import PublicCandidate; k=PublicCandidate()",
                        pipeline="k.pipeline(bytes(range(32)),condition=condition)")
    new = script.format(setup="from keyprint import Keyprint; k=Keyprint(key=bytes(range(32)))",
                        pipeline="k.pipeline(condition=condition)")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "sdk"))
    reference = subprocess.run([sys.executable, "-c", old], env=env, cwd=tmp_path, capture_output=True, text=True)
    port = subprocess.run([sys.executable, "-c", new], cwd=tmp_path, capture_output=True, text=True)
    assert reference.returncode == 0, reference.stderr
    assert port.returncode == 0, port.stderr
    assert json.loads(reference.stdout) == json.loads(port.stdout)


def test_failure_report_retained_without_retry(tmp_path):
    calls = []
    def bad_model(ids, *, cache):
        calls.append(1)
        raise RuntimeError("fixture failure")
    output = tmp_path / "run"
    with pytest.raises(KeyprintError) as error:
        Keyprint(key=KEY)._run(bad_model, [32], max_tokens=4, condition="marked", output=output,
                              backend=FakeBackend, cache_factory=lambda model: [])
    assert calls == [1]
    assert error.value.artifacts == output
    assert json.loads((output / "report.json").read_text())["report"]["kind"] == "error"
    assert (output / "journal.jsonl").stat().st_size > 0
    if os.name == "posix":
        assert output.stat().st_mode & 0o077 == 0


def test_generation_does_not_overwrite_outputs(tmp_path):
    def model(ids, *, cache):
        return head(32).reshape(1, 1, 151936).repeat(ids.shape[1], axis=1)
    keyprint = Keyprint(key=KEY)
    output = tmp_path / "run"
    result = keyprint._run(model, [32], max_tokens=2, condition="ordinary", output=output,
                          backend=FakeBackend, cache_factory=lambda model: [])
    assert result.text == "AA"
    assert result.report["verdict"] is None
    with pytest.raises(FileExistsError):
        keyprint._run(model, [32], max_tokens=2, condition="ordinary", output=output,
                      backend=FakeBackend, cache_factory=lambda model: [])


def test_corruption_rejected(tmp_path):
    import keyprint
    copied = tmp_path / "keyprint"
    shutil.copytree(Path(keyprint.__file__).parent, copied)
    file = copied / "_engine/research/keyprint_reporting_v2.py"
    file.write_bytes(file.read_bytes() + b"\n# disposable corruption\n")
    result = subprocess.run([sys.executable, "-m", "keyprint", "verify"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "integrity mismatch" in result.stderr


def test_keygen_never_prints_or_overwrites_key(tmp_path):
    path = tmp_path / "owner.key"
    command = [sys.executable, "-m", "keyprint", "keygen", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    key = path.read_bytes()
    assert len(key) == 32 and key.hex() not in result.stdout
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert path.read_bytes() == key
    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("command", ["demo", "doctor", "verify"])
def test_cli_json(command, tmp_path):
    result = subprocess.run([sys.executable, "-m", "keyprint", command, "--json"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout), dict)


def test_unavailable_and_unsupported_backend_errors():
    candidate = Keyprint(key=KEY)
    with pytest.raises(ValueError, match="backend"):
        candidate.generate("hello")
    assert verify()["files"] == 39


@pytest.mark.parametrize("cancel_at", [0, 1, 2])
def test_reference_cancel_preserves_consumption_and_releases_candidate(tmp_path, cancel_at):
    stop = Event()
    calls = []
    def model(ids, *, cache):
        calls.append(1)
        if len(calls) == cancel_at:
            stop.set()
        return head(32).reshape(1, 1, 151936).repeat(ids.shape[1], axis=1)
    kp = Keyprint(key=KEY)
    if cancel_at == 0:
        stop.set()
    with pytest.raises(KeyprintCancelled) as caught:
        kp._run(model, [32], max_tokens=4, condition="marked", output=tmp_path / "cancelled",
                backend=FakeBackend, cache_factory=lambda model: [], cancel_event=stop)
    report = caught.value.report
    assert len(calls) == cancel_at
    assert report["payload"]["committed_tokens"] == (1 if cancel_at == 2 else 0)
    assert report["usage"]["completion_tokens"] == (1 if cancel_at == 2 else 0)
    assert report["cancellation_requested"] is True
    assert json.loads((caught.value.artifacts / "report.json").read_text())["report"] == report
    result = kp._run(model, [32], max_tokens=2, condition="marked", output=tmp_path / "next",
                     backend=FakeBackend, cache_factory=lambda model: [])
    assert result.text == "AA"


def test_reference_cancellation_does_not_mask_a_model_failure(tmp_path):
    stop = Event()
    def model(ids, *, cache):
        stop.set()
        raise RuntimeError("actual failure")
    with pytest.raises(KeyprintError) as caught:
        Keyprint(key=KEY)._run(model, [32], max_tokens=4, condition="marked", output=tmp_path / "failed",
                              backend=FakeBackend, cache_factory=lambda model: [], cancel_event=stop)
    assert not isinstance(caught.value, KeyprintCancelled)
    assert "cancellation_requested" not in caught.value.report


def test_reference_cancellation_during_draw_finishes_that_commit(tmp_path, monkeypatch):
    stop = Event()
    calls = []
    def model(ids, *, cache):
        calls.append(1)
        return head(32, 33).reshape(1, 1, 151936).repeat(ids.shape[1], axis=1)
    def draw(bits):
        stop.set()
        return (1 << bits) - 1
    monkeypatch.setattr("keyprint.api.secrets.randbits", draw)
    with pytest.raises(KeyprintCancelled) as caught:
        Keyprint(key=KEY)._run(model, [32], max_tokens=4, condition="ordinary", output=tmp_path / "stopped",
                              backend=FakeBackend, cache_factory=lambda model: [], cancel_event=stop)
    assert calls == [1]
    assert caught.value.report["usage"]["completion_tokens"] == 1
    events = [json.loads(line)["event"] for line in (caught.value.artifacts / "journal.jsonl").read_text().splitlines()]
    kinds = [event["kind"] for event in events]
    assert kinds.index("random_bits_returned") < kinds.index("committed_step")
