"""Explicit native-library wrapper for development parity/timing only."""
import ctypes
from pathlib import Path


class BatchSHA256:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.call = self.library.keyprint_prf_batch
        self.call.argtypes = [ctypes.c_void_p, ctypes.c_size_t] * 3 + [
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t]
        self.call.restype = ctypes.c_int
        version = self.library.keyprint_prf_openssl_version
        version.restype = ctypes.c_char_p
        version.argtypes = []
        self.openssl_version = version().decode('ascii')

    def digests(self, key, prefix, suffixes, layers=30):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError('exactly 32 key bytes required')
        if type(prefix) is not bytes or len(prefix) > 65536:
            raise ValueError('prefix must be bytes up to 65536 bytes')
        if type(layers) is not int or not 1 <= layers <= 64:
            raise ValueError('layers must be an integer from 1 to 64')
        if type(suffixes) not in (list, tuple) or not 1 <= len(suffixes) <= 1000:
            raise ValueError('supply 1 to 1000 suffixes')
        offsets = [0]
        for suffix in suffixes:
            if type(suffix) is not bytes or len(suffix) > 4096:
                raise ValueError('suffixes must be bytes up to 4096 bytes')
            offsets.append(offsets[-1] + len(suffix))
        if offsets[-1] > 1048576:
            raise ValueError('combined suffixes exceed 1048576 bytes')
        data = b''.join(suffixes)
        sizes = (ctypes.c_size_t * len(offsets))(*offsets)
        result = ctypes.create_string_buffer(len(suffixes) * layers * 32)
        if self.call(key, len(key), prefix, len(prefix), data, len(data),
                     sizes, len(offsets), layers, result, len(result)) != 1:
            raise RuntimeError('native HMAC batch failed; no partial result accepted')
        # Return every byte for parity auditing; first-bit extraction stays in Python.
        return result.raw
