import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
repo=Path.cwd();out=Path(__file__).resolve().parent
worker_env=os.environ.copy();worker_env.pop('PYTHONPATH',None)
worker_env.update(UV_OFFLINE='1',UV_CONCURRENT_BUILDS='1',UV_CONCURRENT_INSTALLS='1',UV_CONCURRENT_DOWNLOADS='1')
steps=[]
def run(name,cmd,cwd,expected=0):
 p=subprocess.run(cmd,cwd=cwd,env=worker_env,text=True,capture_output=True,timeout=90)
 (out/f'{name}.stdout').write_text(p.stdout);(out/f'{name}.stderr').write_text(p.stderr)
 steps.append(dict(step=name,exit_code=p.returncode));print(json.dumps(steps[-1]),flush=True)
 assert p.returncode==expected,(name,p.returncode,p.stderr[-1000:])
 return p.stdout
uv='/Users/vincent/.local/bin/uv'
run('build',[uv,'build','--offline','--wheel','--out-dir',str(out/'dist')],repo)
wheel=next((out/'dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
 names=z.namelist()
 assets=[p for p in (repo/'src/keyprint/web').iterdir() if p.is_file()]
 assert all(z.read('keyprint/web/'+p.name)==p.read_bytes() for p in assets)
 assert not any('/tests/' in n or n.startswith('tools/') for n in names)
 metadata=z.read('keyprint-0.1.0a1.dist-info/METADATA').decode()
 assert 'Name: keyprint\n' in metadata and 'Version: 0.1.0a1\n' in metadata
run('venv',[uv,'venv','--offline','--python',sys.executable,str(out/'env')],out)
python=str(out/'env/bin/python');cli=str(out/'env/bin/keyprint')
run('install',[uv,'pip','install','--offline','--python',python,str(wheel)],out)
run('dependency-check',[uv,'pip','check','--python',python],out)
source_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'src/keyprint').rglob('*.py')}
identity=json.loads(run('import',[python,'-I','-c','import keyprint,json,sys;print(json.dumps({"module":keyprint.__file__,"python":sys.executable}))'],out))
assert Path(identity['module']).is_relative_to(out/'env')
run('footprint',[python,'-I',str(out/'measure_core.py')],out)
for command in ('doctor','verify','demo'):
 result=run(command,[cli,command,'--json'],out)
 data=json.loads(result)
 if command!='demo':assert data['status']=='pass'
run('help',[cli,'--help'],out)
installed=json.loads(run('packages',[uv,'pip','list','--python',python,'--format','json'],out))
assert not ({'torch','mlx','mlx-lm','transformers','fastapi','openai','anthropic'} & {r['name'] for r in installed})
summary=dict(source_sha256=source_hashes,wheel=wheel.name,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),wheel_bytes=wheel.stat().st_size,asset_count=len(assets),all_assets_exact=True,isolated_import=True,dependencies_checked=True,backend_packages_installed=False,model_loaded=False,package_resolution_offline=True,network_activity_audited=False,steps=steps,scientific_acceptance=False,launch_ready=False)
(out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
