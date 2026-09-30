import ast,json,os,subprocess,sys
from pathlib import Path
repo=Path.cwd();out=Path(__file__).resolve().parent
module=ast.parse((repo/'tests/test_public_api.py').read_text())
fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='test_reference_parity')
script=next(ast.literal_eval(n.value) for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
checks=[]
for condition in ('ordinary','marked'):
 for seed in range(8):
  data=[]
  for arm in ('preserved','current'):
   setup='from keyprint_v3 import PublicCandidate; k=PublicCandidate()' if arm=='preserved' else 'from keyprint import Keyprint; k=Keyprint(key=bytes(range(32)))'
   pipeline='k.pipeline(bytes(range(32)),condition=condition)' if arm=='preserved' else 'k.pipeline(condition=condition)'
   code=script.format(setup=setup,pipeline=pipeline).replace("for condition in ('ordinary', 'marked'):",f'for condition in ({condition!r},):').replace('for seed in range(8):',f'for seed in ({seed},):')
   file=out/f'parity-{condition}-{seed}-{arm}.py';file.write_text(code)
   guard=out/f'parity-{condition}-{seed}-{arm}'
   env=dict(os.environ,PYTHONPATH=str(repo/('sdk' if arm=='preserved' else 'src')))
   result=subprocess.run([sys.executable,str(repo/'tools/memory_watchdog.py'),'--output',str(guard),'--limit-gib','0.5','--timeout','60','--',sys.executable,str(file)],env=env,cwd=out,text=True,capture_output=True)
   receipt=json.loads((guard/'result.json').read_text())
   if result.returncode:
    print(json.dumps(dict(condition=condition,seed=seed,arm=arm,resource=receipt)),flush=True);sys.exit(1)
   data.append(json.loads((guard/'worker.log').read_text()))
  assert data[0]==data[1],(condition,seed)
  checks.append(dict(condition=condition,seed=seed,steps=16,exact_match=True))
  (out/'parity.json').write_text(json.dumps(dict(checks=checks,complete=len(checks)==16),indent=2)+'\n')
  print(json.dumps(checks[-1]),flush=True)
