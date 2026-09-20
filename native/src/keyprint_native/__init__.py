"""Optional binary accelerator. No model, network access or persistent keyed state."""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import platform

__version__ = '0.1.0a2'


class NativePRF:
    """Load only the bundled, hash-checked library; never discover system crypto."""
    def __init__(self):
        root=Path(__file__).parent
        manifest_path=root/'native-build.json'
        try:
            manifest=json.loads(manifest_path.read_text())
            expected={'_native.c','_select.c','__init__.py','licenses/OpenSSL.txt','_native.dylib'}
            if (manifest['schema']!='keyprint.native-build.v1' or manifest['package_version']!=__version__
                    or set(manifest['files'])!=expected or manifest['architecture']!='arm64'
                    or manifest['dynamic_dependencies']!=['/usr/lib/libSystem.B.dylib']):
                raise ValueError('unexpected native manifest')
            for name,digest in manifest['files'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'native file integrity mismatch: {name}')
            if platform.system()!='Darwin' or platform.machine()!='arm64':
                raise RuntimeError('This native wheel requires Apple Silicon macOS')
            minimum=tuple(map(int,manifest['minimum_macos'].split('.')))
            current=tuple(map(int,platform.mac_ver()[0].split('.')))
            if current<minimum:
                raise RuntimeError('This native wheel requires macOS '+manifest['minimum_macos']+' or newer')
        except (OSError,KeyError,TypeError,ValueError) as exc:
            raise RuntimeError('Native package is incomplete or fails integrity checks; reinstall its exact wheel') from exc
        self._library=ctypes.CDLL(str(root/'_native.dylib'))
        self._call=self._library.keyprint_prf_batch
        self._call.argtypes=[ctypes.c_void_p,ctypes.c_size_t]*3+[
            ctypes.POINTER(ctypes.c_size_t),ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        self._call.restype=ctypes.c_int
        self._select=self._library.keyprint_select_f32
        self._select.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,
                              ctypes.c_void_p,ctypes.c_size_t]
        self._select.restype=ctypes.c_int
        version=self._library.keyprint_prf_openssl_version
        version.argtypes=[]
        version.restype=ctypes.c_char_p
        self._identity={'schema':'keyprint.native-prf.v1','package_version':__version__,
                        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        'build':manifest,'openssl_version':version().decode('ascii'),
                        'selection':{'encoding':'little-endian-binary32',
                                     'order':'descending score, ascending original index; signed zeros tie',
                                     'maximum_scores':151669,'maximum_top_k':512},
                        'persistent_key_cache':False,'empirical_acceptance_transfers':False}

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def digests(self,key,prefix,suffixes,layers=30):
        if type(key) is not bytes or len(key)!=32:
            raise ValueError('exactly 32 key bytes required')
        if type(prefix) is not bytes or len(prefix)>65536:
            raise ValueError('prefix must be bytes up to 65536 bytes')
        if type(layers) is not int or not 1<=layers<=64:
            raise ValueError('layers must be an integer from 1 to 64')
        if type(suffixes) not in (list,tuple) or not 1<=len(suffixes)<=1000:
            raise ValueError('supply 1 to 1000 suffixes')
        offsets=[0]
        for suffix in suffixes:
            if type(suffix) is not bytes or len(suffix)>4096:
                raise ValueError('suffixes must be bytes up to 4096 bytes')
            offsets.append(offsets[-1]+len(suffix))
        if offsets[-1]>1048576:
            raise ValueError('combined suffixes exceed 1048576 bytes')
        data=b''.join(suffixes)
        sizes=(ctypes.c_size_t*len(offsets))(*offsets)
        result=ctypes.create_string_buffer(len(suffixes)*layers*32)
        if self._call(key,len(key),prefix,len(prefix),data,len(data),sizes,len(offsets),layers,result,len(result))!=1:
            raise RuntimeError('native HMAC batch failed; no partial result accepted')
        return result.raw

    def select_indices(self, scores, top_k):
        """Finite binary32 byte scores; immutable positions in descending order."""
        if (type(scores) is not bytes or len(scores)%4 or not 4<=len(scores)<=151669*4
                or type(top_k) is not int or not 1<=top_k<=min(512,len(scores)//4)):
            raise ValueError('bounded binary32 bytes and integer top-k from 1 to 512 required')
        output=(ctypes.c_uint32*top_k)()
        if self._select(scores,len(scores),top_k,output,ctypes.sizeof(output))!=1:
            raise RuntimeError('native selection failed; no partial result accepted')
        return tuple(output)
