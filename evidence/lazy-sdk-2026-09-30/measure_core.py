import ctypes,json,os,sys
from pathlib import Path
class Usage(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in ('user_time','system_time','idle_wakeups','interrupt_wakeups','pageins','wired','resident','footprint','started','exited')]
lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
lib.proc_pid_rusage.restype=ctypes.c_int
samples=[]
def sample(stage):
    value=Usage()
    if lib.proc_pid_rusage(os.getpid(),0,ctypes.byref(value)):raise OSError(ctypes.get_errno(),'footprint unavailable')
    samples.append(dict(stage=stage,physical_footprint_bytes=value.footprint,resident_bytes=value.resident))
sample('python_baseline')
from keyprint import Keyprint
sample('import_keyprint')
instances=[Keyprint(key=bytes(32)) for _ in range(20)]
assert all(x._reference_candidate is None for x in instances)
sample('twenty_unbound_instances')
assert not any(n=='torch' or n=='mlx' or n=='transformers' for n in sys.modules)
for sdk in instances:sdk.close()
sample('closed_unbound_instances')
result=dict(samples=samples,model_loaded=False,reference_tokenizer_loaded=False,inference_measured=False,scope='Point samples in one isolated core-only macOS process; not a peak or a production-inference overhead estimate.')
Path(__file__).with_name('core-footprint.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
