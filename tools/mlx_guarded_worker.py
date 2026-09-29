"""Research-only MLX resource policy applied before loading a model or runner.

Run under memory_watchdog.py. The allocator guideline is NOT the enforced cutoff.
No model downloads or inference are performed by this wrapper itself.
"""
import gc
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import threading
import time


def main():
    parent = int(os.environ.get('KEYPRINT_WATCHDOG_PID', '0'))
    if len(sys.argv) < 2 or 'KEYPRINT_MLX_MEMORY_LOG' not in os.environ or os.getppid() != parent:
        raise RuntimeError('Run a target script through the external memory watchdog')
    import mlx.core as mx
    log_path = Path(os.environ['KEYPRINT_MLX_MEMORY_LOG'])
    previous = {'cache': mx.set_cache_limit(0), 'memory_guideline': mx.set_memory_limit(8 * 1024**3)}
    mx.clear_cache()
    done = threading.Event()
    with log_path.open('x') as log:
        def sample():
            log.write(json.dumps({'time': time.time(), 'active_bytes': mx.get_active_memory(),
                'cache_bytes': mx.get_cache_memory(), 'peak_bytes': mx.get_peak_memory()})+'\n')
            log.flush()
        log.write(json.dumps({'policy': 'cache-disabled-v1', 'cache_limit_bytes': 0,
            'memory_guideline_bytes': 8 * 1024**3, 'previous': previous,
            'enforcement': 'External watchdog, not MLX memory guideline'})+'\n')
        sample()
        def telemetry():
            try:
                while not done.wait(.5):
                    if os.getppid() != parent: os._exit(125)
                    sample()
            except BaseException:
                # No silent telemetry failure while heavy work continues.
                os.kill(os.getpid(), signal.SIGTERM)
        thread = threading.Thread(target=telemetry, daemon=True)
        thread.start()
        try:
            target, sys.argv = sys.argv[1], sys.argv[1:]
            sys.path.insert(0, str(Path(target).resolve().parent))
            runpy.run_path(target, run_name='__main__')
        finally:
            done.set(); thread.join(timeout=2)
            mx.synchronize(); gc.collect(); mx.clear_cache()
            sample()


if __name__ == '__main__': main()
