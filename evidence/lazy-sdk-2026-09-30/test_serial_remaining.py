import json, os, subprocess, sys
from pathlib import Path
repo=Path.cwd(); out=Path(__file__).resolve().parent
python=sys.executable
env=dict(os.environ,PYTHONPATH=str(repo/'src')+os.pathsep+str(repo/'tools'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
collection=subprocess.run([python,'-m','pytest','--collect-only','-q','tests/test_public_api.py'],env=env,text=True,capture_output=True,check=True)
(out/'collection.log').write_text(collection.stdout)
nodes=[line for line in collection.stdout.splitlines() if line.startswith('tests/') and '::' in line]
assert nodes
steps=json.loads((out/"public-tests.json").read_text())
for index,node in list(enumerate(nodes))[10:]:
 path=out/f'public-{index:02}'
 result=subprocess.run([python,'tools/memory_watchdog.py','--output',str(path),'--limit-gib','0.5','--timeout','180','--',python,'-m','pytest','-q',node],env=env,text=True,capture_output=True)
 receipt=json.loads((path/'result.json').read_text())
 steps.append(dict(node=node,exit_code=result.returncode,resource=receipt,pytest=(path/'worker.log').read_text() if (path/'worker.log').exists() else 'not started'))
 (out/'public-tests.json').write_text(json.dumps(steps,indent=2)+'\n')
 print(json.dumps(dict(index=index,node=node,status=receipt['status'],exit=result.returncode)),flush=True)
 if result.returncode:sys.exit(result.returncode)
