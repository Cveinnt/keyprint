"""Explicit source build; wheels contain their private, statically linked crypto."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildNative(build_py):
    def run(self):
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise RuntimeError('Only the macOS ARM64 wheel builder is currently qualified')
        root_value = os.environ.get('KEYPRINT_OPENSSL_ROOT')
        if not root_value:
            raise RuntimeError('Source build requires KEYPRINT_OPENSSL_ROOT pointing to an explicit OpenSSL 3 static build; install a prebuilt wheel instead')
        root = Path(root_value).resolve()
        archive = root/'lib/libcrypto.a'
        if not archive.is_file():
            raise RuntimeError('Explicit OpenSSL root must contain lib/libcrypto.a')
        target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', '11.0')
        if not re.fullmatch(r'\d+\.\d+', target):
            raise RuntimeError('MACOSX_DEPLOYMENT_TARGET must be major.minor')
        archive_commands = subprocess.check_output(['otool','-l',str(archive)],text=True)
        archive_versions = re.findall(r'\bminos\s+([\d.]+)',archive_commands)
        if not archive_versions:
            raise RuntimeError('Could not verify the static archive minimum OS')
        version_tuple = lambda value: tuple((list(map(int,value.split('.')))+[0,0,0])[:3])
        archive_minimum = max(archive_versions,key=version_tuple)
        if version_tuple(target)<version_tuple(archive_minimum):
            raise RuntimeError(f'OpenSSL objects require macOS {archive_minimum}; cannot label them for {target}')
        super().run()
        package = Path(self.build_lib)/'keyprint_native'
        binary = package/'_native.dylib'
        source = Path('src/keyprint_native/_native.c')
        command = ['clang','-std=c11','-O3','-Wall','-Wextra','-Werror','-dynamiclib',
                   '-arch','arm64',f'-mmacosx-version-min={target}',str(source),
                   'src/keyprint_native/_select.c',
                   '-I'+str(root/'include'),str(archive),'-o',str(binary),
                   '-Wl,-dead_strip','-Wl,-install_name,@rpath/_native.dylib']
        subprocess.run(command, check=True)
        linked = subprocess.check_output(['otool','-L',str(binary)],text=True)
        dependencies = [line.strip().split(' (')[0] for line in linked.splitlines()[1:]]
        if dependencies != ['@rpath/_native.dylib','/usr/lib/libSystem.B.dylib']:
            raise RuntimeError(f'Unexpected dynamic dependency: {dependencies}')
        load_commands = subprocess.check_output(['otool','-l',str(binary)],text=True)
        minimums = re.findall(r'\bminos\s+([\d.]+)',load_commands)
        if len(minimums)!=1:
            raise RuntimeError('Could not determine the binary minimum macOS version')
        arch = subprocess.check_output(['lipo','-archs',str(binary)],text=True).strip()
        if arch!='arm64':
            raise RuntimeError('Native binary architecture mismatch')
        files=['_native.c','_select.c','__init__.py','licenses/OpenSSL.txt','_native.dylib']
        receipt={'schema':'keyprint.native-build.v1','package_version':'0.1.0a2',
                 'architecture':arch,'minimum_macos':minimums[0],
                 'openssl_object_minimum_macos':archive_minimum,
                 'dynamic_dependencies':dependencies[1:],
                 'openssl_static_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                 'files':{name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in files},
                 'compiler':subprocess.check_output(['clang','--version'],text=True).splitlines()[0],
                 'scope':'Experimental; build identity is not authenticity or scientific acceptance'}
        (package/'native-build.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        build = self.get_finalized_command('build_py')
        receipt=json.loads((Path(build.build_lib)/'keyprint_native/native-build.json').read_text())
        major,minor,*_=map(int,receipt['minimum_macos'].split('.'))
        if major>=11: minor=0
        return 'py3','none',f'macosx_{major}_{minor}_arm64'


setup(distclass=NativeDistribution, cmdclass={'build_py':BuildNative,'bdist_wheel':PlatformWheel})
