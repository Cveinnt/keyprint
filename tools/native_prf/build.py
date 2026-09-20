"""Explicit local build of the development helper; never runs during pip install."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New private build directory')
    parser.add_argument('--compiler', default='clang')
    args = parser.parse_args()
    system = platform.system()
    if system not in ('Darwin', 'Linux'):
        parser.error('this development builder supports macOS and Linux only')
    # Probe explicit local prerequisites before creating output. Never download.
    version = subprocess.check_output(['pkg-config', '--modversion', 'openssl'], text=True).strip()
    if not version.startswith('3.'):
        parser.error('OpenSSL 3 development headers/libraries are required')
    flags = shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'openssl'], text=True))
    compiler = subprocess.check_output([args.compiler, '--version'], text=True)
    args.output.mkdir(mode=0o700)
    source = Path(__file__).with_name('batch_sha256.c').resolve()
    library = (args.output / ('libkeyprint_prf.dylib' if system == 'Darwin' else 'libkeyprint_prf.so')).resolve()
    command = [args.compiler, '-std=c11', '-O3', '-Wall', '-Wextra', '-Werror']
    command += ['-dynamiclib'] if system == 'Darwin' else ['-shared', '-fPIC']
    command += [str(source), '-o', str(library), *flags]
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output/'build.log').write_text(result.stdout+result.stderr)
    receipt = {'command': command, 'compiler': compiler, 'openssl_pkg_config_version': version,
               'system': system, 'machine': platform.machine(), 'returncode': result.returncode,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest() if result.returncode == 0 else None}
    (args.output/'build.json').write_text(json.dumps(receipt, indent=2))
    if result.returncode:
        print(f'Build failed. Retained log: {args.output / "build.log"}')
        return 1
    print(library)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
