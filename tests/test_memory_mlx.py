"""Small real Metal allocations only: no model loads, downloads or inference."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from memory_watchdog import supervise

pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin' or importlib.util.find_spec('mlx') is None,
    reason='Requires macOS and optional MLX runtime')


def test_real_gpu_allocation_visible_and_cache_released(tmp_path):
    target = tmp_path/'probe.py'
    target.write_text('''import gc, json, os, time
import mlx.core as mx
from memory_watchdog import MacMemory
monitor = MacMemory()
before = monitor.usage(os.getpid())['footprint_bytes']
x = mx.ones((16*1024*1024,), dtype=mx.float32)
mx.eval(x)
after = monitor.usage(os.getpid())['footprint_bytes']
assert mx.get_active_memory() >= 64*1024**2
assert after-before >= 32*1024**2
assert mx.get_cache_memory() == 0
print(json.dumps({'before': before, 'after': after}), flush=True)
time.sleep(.8)
del x
gc.collect(); mx.synchronize(); mx.clear_cache()
assert mx.get_active_memory() == 0
assert mx.get_cache_memory() == 0
''')
    r = supervise([sys.executable, str(ROOT/'tools/mlx_guarded_worker.py'), str(target)],
        tmp_path/'run', limit_bytes=512*1024**2, timeout=10, disk_floor=0)
    assert r['status'] == 'completed' and r['cleanup_verified']
    events = [json.loads(s) for s in (tmp_path/'run/mlx-memory.jsonl').read_text().splitlines()]
    assert events[0]['cache_limit_bytes'] == 0
    assert any(e.get('active_bytes', 0) >= 64*1024**2 for e in events)
    assert events[-1]['active_bytes'] == events[-1]['cache_bytes'] == 0


def test_guarded_mlx_worker_exits_if_watchdog_is_killed(tmp_path):
    target = tmp_path/'wait.py'
    target.write_text('import time; print("ready", flush=True); time.sleep(20)')
    out = tmp_path/'run'
    command = [sys.executable, str(ROOT/'tools/memory_watchdog.py'), '--output', str(out),
        '--limit-gib', '.5', '--timeout', '10', '--', sys.executable,
        str(ROOT/'tools/mlx_guarded_worker.py'), str(target)]
    guardian = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    try:
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            log = out/'worker.log'
            if log.exists() and 'ready' in log.read_text():
                samples = (out/'memory.jsonl').read_text().splitlines()
                if samples:
                    child = json.loads(samples[-1])['processes'][0]['pid']
                    break
            time.sleep(.05)
        else: pytest.fail('Guarded MLX worker did not start')
        guardian.kill(); guardian.wait(timeout=3)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = subprocess.run(['ps', '-p', str(child), '-o', 'stat='],
                capture_output=True, text=True).stdout.strip()
            if not state or state.startswith('Z'): break
            time.sleep(.05)
        else: pytest.fail('MLX worker survived watchdog loss')
        assert not (out/'result.json').exists()  # SIGKILL is an interrupted attempt, never success.
    finally:
        if guardian.poll() is None:
            guardian.terminate(); guardian.wait(timeout=5)
        if child is not None:
            try: os.kill(child, signal.SIGKILL)
            except ProcessLookupError: pass
