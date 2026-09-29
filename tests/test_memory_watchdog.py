import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
from memory_watchdog import MacMemory, supervise

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS physical footprint')


def run(tmp_path, code, **kwargs):
    return supervise([sys.executable, '-c', code], tmp_path/'run',
        interval=.05, disk_floor=0, timeout=5, **kwargs)


def test_native_measurement_includes_process_identity():
    r = MacMemory().usage(os.getpid())
    assert r['pid'] == os.getpid() and r['footprint_bytes'] > 0 and r['started'] > 0


def test_normal_worker_retains_output_and_final_receipt(tmp_path):
    result = run(tmp_path, 'import time; print("retained"); time.sleep(.15)')
    assert result['status'] == 'completed' and result['exit_code'] == 0
    assert result['samples'] > 0
    assert (tmp_path/'run/worker.log').read_text().strip() == 'retained'
    assert json.loads((tmp_path/'run/result.json').read_text()) == result


def test_budget_actually_kills_worker_even_when_term_ignored(tmp_path):
    result = run(tmp_path, 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        'x=bytearray(96*1024**2); time.sleep(20)', limit_bytes=64*1024**2)
    assert result['status'] == 'stopped' and result['reason'] == 'memory_budget'
    assert result['peak_footprint_bytes'] > 64*1024**2
    assert result['exit_code'] == -signal.SIGKILL
    with pytest.raises(ProcessLookupError): os.kill(result['pid'], 0)


def test_pressure_preflight_prevents_worker_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(MacMemory, 'pressure', lambda self: 2)
    r = run(tmp_path, 'raise Exception("must not execute")')
    assert r['status'] == 'monitor_failed' and 'pid' not in r
    assert 'System memory pressure' in r['error']


def test_failed_sensor_stops_worker_instead_of_silently_unmonitoring(tmp_path, monkeypatch):
    def fail(self, pid): raise OSError('sensor failure')
    monkeypatch.setattr(MacMemory, 'usage', fail)
    r = run(tmp_path, 'import time; time.sleep(20)')
    assert r['status'] == 'monitor_failed' and r['exit_code'] < 0
    with pytest.raises(ProcessLookupError): os.kill(r['pid'], 0)


def test_deadline_stops_worker(tmp_path):
    r = supervise([sys.executable, '-c', 'import time; time.sleep(20)'], tmp_path/'run',
        interval=.05, disk_floor=0, timeout=.15)
    assert r['reason'] == 'time_budget' and r['exit_code'] < 0


def test_pressure_during_run_stops_worker(tmp_path, monkeypatch):
    readings = iter([1, 2])
    monkeypatch.setattr(MacMemory, 'pressure', lambda self: next(readings))
    r = run(tmp_path, 'import time; time.sleep(20)')
    assert r['reason'] == 'system_pressure' and r['cleanup_verified']


def test_low_disk_during_run_stops_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import memory_watchdog
    readings = iter([SimpleNamespace(free=100), SimpleNamespace(free=-1)])
    monkeypatch.setattr(memory_watchdog.shutil, 'disk_usage', lambda path: next(readings))
    r = run(tmp_path, 'import time; time.sleep(20)')
    assert r['reason'] == 'disk_headroom' and r['cleanup_verified']


def test_cleanup_failure_retains_receipt_and_is_never_success(tmp_path, monkeypatch):
    import memory_watchdog
    original = memory_watchdog.stop
    def fail_after_cleanup(process):
        original(process)
        raise PermissionError('fixture cleanup failure')
    monkeypatch.setattr(memory_watchdog, 'stop', fail_after_cleanup)
    r = run(tmp_path, 'pass')
    assert r['status'] == 'cleanup_failed' and not r['cleanup_verified']
    assert r['cleanup_error'] == 'fixture cleanup failure'
    assert json.loads((tmp_path/'run/result.json').read_text()) == r


def test_exited_leader_does_not_leave_term_ignoring_child(tmp_path):
    child = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(20)'
    code = f'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",{child!r}],stdout=subprocess.PIPE,text=True); p.stdout.readline(); print(p.pid,flush=True)'
    r = run(tmp_path, code)
    assert r['status'] == 'completed' and r['cleanup_verified']
    child_pid = int((tmp_path/'run/worker.log').read_text().strip())
    state = subprocess.run(['ps','-p',str(child_pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
    assert not state or state.startswith('Z')


def test_process_group_memory_counts_children_and_stops_them(tmp_path):
    child = 'import time; x=bytearray(48*1024**2); time.sleep(20)'
    code = f'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",{child!r}]); print(p.pid,flush=True); x=bytearray(48*1024**2); time.sleep(20)'
    r = run(tmp_path, code, limit_bytes=96*1024**2)
    assert r['reason'] == 'memory_budget'
    samples = [json.loads(s) for s in (tmp_path/'run/memory.jsonl').read_text().splitlines()]
    assert any(len(s['processes']) == 2 for s in samples)
    child_pid = int((tmp_path/'run/worker.log').read_text().strip())
    state = subprocess.run(['ps','-p',str(child_pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
    assert not state or state.startswith('Z')


def test_cli_cancellation_cleans_up_child(tmp_path):
    root = Path(__file__).parents[1]
    out = tmp_path/'run'
    command = [sys.executable,str(root/'tools/memory_watchdog.py'),'--output',str(out),
        '--',sys.executable,'-c','import time; time.sleep(20)']
    p = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            f=out/'memory.jsonl'
            if f.exists() and f.stat().st_size: break
            time.sleep(.02)
        else: pytest.fail('Watchdog did not start')
        p.terminate(); p.wait(timeout=5)
        r=json.loads((out/'result.json').read_text())
        assert r['status']=='cancelled' and r['exit_code']<0
        with pytest.raises(ProcessLookupError): os.kill(r['pid'],0)
    finally:
        if p.poll() is None: p.kill(); p.wait()


def test_existing_receipt_directory_cannot_be_reused(tmp_path):
    (tmp_path/'run').mkdir()
    with pytest.raises(FileExistsError): run(tmp_path, 'pass')


def test_another_guarded_run_cannot_stack_memory_budgets(tmp_path):
    import fcntl
    lock_dir = Path.home()/'.cache/keyprint'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/'research-worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run(tmp_path, 'raise Exception("must not execute")')
    assert r['status'] == 'monitor_failed' and 'pid' not in r
    assert r['error_type'] == 'BlockingIOError'


@pytest.mark.parametrize('limit', [0, 11*1024**3])
def test_invalid_memory_budget_never_starts_worker(tmp_path,limit):
    with pytest.raises(ValueError): run(tmp_path, 'pass',limit_bytes=limit)
