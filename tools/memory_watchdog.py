"""macOS research-process supervisor; sampled GPU-inclusive cutoff, not an OS quota.

Launch only a foreground worker whose descendants retain its process group.
Logs measurements to disk, never captures worker output in memory. No restart.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil


class Cancelled(BaseException):
    pass


class Usage(ctypes.Structure):
    # macOS sys/resource.h rusage_info_v0, including compressed/Metal footprint.
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        'user_time', 'system_time', 'idle_wakeups', 'interrupt_wakeups', 'pageins',
        'wired', 'resident', 'footprint', 'started', 'exited')]


class MacMemory:
    def __init__(self):
        if sys.platform != 'darwin':
            raise RuntimeError('This GPU-inclusive monitor requires macOS')
        self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.proc_pid_rusage.restype = ctypes.c_int
        self.sys = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        self.sys.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
        self.sys.sysctlbyname.restype = ctypes.c_int

    def usage(self, pid):
        value = Usage()
        if self.lib.proc_pid_rusage(pid, 0, ctypes.byref(value)) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot read process footprint')
        return {'pid': pid, 'footprint_bytes': value.footprint,
                'resident_bytes': value.resident, 'started': value.started}

    def pressure(self):
        value, size = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if self.sys.sysctlbyname(b'kern.memorystatus_vm_pressure_level',
                ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot read system memory pressure')
        if value.value not in (1, 2, 4):
            raise RuntimeError('Unknown memory pressure level')
        return value.value


def members(group):
    result = subprocess.run(['ps', '-axo', 'pid=,pgid=,stat='], check=True,
        capture_output=True, text=True, timeout=3)
    return [int(pid) for pid, pgid, state in (line.split() for line in result.stdout.splitlines())
            if int(pgid) == group and not state.startswith('Z')]


def stop(process):
    # Only the fresh process group owned by this supervisor. TERM, then KILL,
    # including descendants if the group leader exits first or ignores TERM.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        process.poll()  # Reap leader: macOS can return EPERM for zombie-only groups.
        if not members(process.pid): break
        try: os.killpg(process.pid, sig)
        except ProcessLookupError: break
        except PermissionError:
            # Processes can exit between enumeration and signaling. Only tolerate
            # an empty/zombie-only group; real permission failures remain errors.
            process.poll()
            if members(process.pid): raise
            break
        if sig == signal.SIGTERM: time.sleep(.5)
    process.wait(timeout=5)
    deadline = time.monotonic() + 2
    while members(process.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError('Worker process group did not stop')
        time.sleep(.05)


def supervise(command, output, *, limit_bytes=10 * 1024**3, interval=.25,
              timeout=1800., disk_floor=2 * 1024**3):
    if (not command or not 16 * 1024**2 <= limit_bytes <= 10 * 1024**3
            or not .05 <= interval <= 1 or not 0 < timeout <= 3600 or disk_floor < 0):
        raise ValueError('Invalid bounded-run policy')
    output.mkdir(mode=0o700)  # never overwrite or silently resume a prior run
    result = {'status': 'not_started', 'limit_bytes': limit_bytes, 'interval_seconds': interval,
        'timeout_seconds': timeout, 'disk_floor_bytes': disk_floor,
        'peak_footprint_bytes': 0, 'samples': 0, 'automatic_restart': False,
        'scope': 'Sum of process-group physical footprints; shared pages may be double-counted. Sampled termination allows transient overshoot, not an OS allocation quota.'}
    process = None
    lock = None
    began = time.monotonic()
    try:
        monitor = MacMemory()
        import fcntl
        lock_dir = Path.home()/'.cache/keyprint'
        lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = os.open(lock_dir/'research-worker.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        # A second guarded research run fails closed instead of doubling budgets.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if monitor.pressure() != 1: raise RuntimeError('System memory pressure before launch')
        if shutil.disk_usage(output).free < disk_floor: raise RuntimeError('Insufficient disk headroom')
        with (output/'worker.log').open('x') as log, (output/'memory.jsonl').open('x') as measurements:
            env = dict(os.environ, KEYPRINT_MLX_MEMORY_LOG=str((output/'mlx-memory.jsonl').resolve()),
                KEYPRINT_WATCHDOG_PID=str(os.getpid()))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env)
            result['pid'] = process.pid
            while process.poll() is None:
                readings = []
                for pid in members(process.pid):
                    try: readings.append(monitor.usage(pid))
                    except OSError:
                        if pid in members(process.pid): raise
                pressure = monitor.pressure()
                total = sum(r['footprint_bytes'] for r in readings)
                free = shutil.disk_usage(output).free
                elapsed = time.monotonic() - began
                measurements.write(json.dumps({'seconds': elapsed, 'processes': readings,
                    'footprint_bytes': total, 'pressure': pressure, 'disk_free_bytes': free})+'\n')
                measurements.flush()
                result['samples'] += 1
                result['peak_footprint_bytes'] = max(result['peak_footprint_bytes'], total)
                reason = ('memory_budget' if total > limit_bytes else
                          'system_pressure' if pressure != 1 else
                          'disk_headroom' if free < disk_floor else
                          'time_budget' if elapsed > timeout else None)
                if reason:
                    result.update(status='stopped', reason=reason)
                    break
                time.sleep(interval)
            if result['status'] != 'stopped':
                result['status'] = 'completed' if process.returncode == 0 else 'worker_failed'
    except Cancelled:
        result.update(status='cancelled', reason='supervisor_signal')
    except BaseException as exc:
        result.update(status='monitor_failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        if process is not None:
            try:
                stop(process)
                result['cleanup_verified'] = True
            except BaseException as exc:
                result['outcome_before_cleanup'] = result['status']
                result.update(status='cleanup_failed', cleanup_verified=False,
                    cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
            result['exit_code'] = process.returncode
        result['elapsed_seconds'] = time.monotonic() - began
        try:
            (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        finally:
            if lock is not None: os.close(lock)
    return result


def main():
    def cancel(signum, frame): raise Cancelled()
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit-gib', type=float, default=10)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    result = supervise(cmd, a.output, limit_bytes=int(a.limit_gib * 1024**3), timeout=a.timeout)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
