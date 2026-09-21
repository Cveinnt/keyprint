if not __debug__:
    raise RuntimeError('run without -O; assertions are required')
from pathlib import Path
import argparse,hashlib,json
from keyprint.backends.bytelevel import ByteLevelBinding
parser=argparse.ArgumentParser(description='Audit native HTTP outputs against private selected tokens and returned IDs')
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--assets',type=Path,required=True)
parser.add_argument('--attempt',default='attempt-1')
args=parser.parse_args()
root=args.root
public=root/'public'/args.attempt
assets=args.assets;config=json.loads((assets/'config.json').read_text());data=json.loads((assets/'tokenizer.json').read_text())
eos=config['eos_token_id']; eos=eos if isinstance(eos,list) else [eos]
binding=ByteLevelBinding.create((assets/'tokenizer.json').read_text(),vocabulary_size=config['vocab_size'],special_ids=[x['id'] for x in data['added_tokens'] if x.get('special')],eos_ids=eos)
traces={}
for p in (root/'traces').glob('*.jsonl'):
 previous='0'*64;ids=[]
 for i,line in enumerate(p.read_bytes().splitlines(keepends=True)):
  value=json.loads(line);assert value['sequence']==i and value['previous_sha256']==previous
  previous=hashlib.sha256(line).hexdigest();event=value['event'];assert event['phase']!='host_prefix_mismatch'
  if event['phase']=='selected_tentative':ids.append(event['token_id'])
 traces[p.name]=ids
r=json.loads((public/'results.json').read_text());checked=[]
for row in r['outputs']:
 resp=row['response']
 if row['protocol']=='openai':
  ids=resp['choices'][0]['token_ids'];matches=[n for n,t in traces.items() if t==ids];assert matches
  text=resp['choices'][0]['message']['content'];count=resp['usage']['completion_tokens']
 else:
  ids=traces[row['journal']];text=''.join(c['text'] for c in resp['content'] if c['type']=='text');count=resp['usage']['output_tokens']
 assert binding.render(ids)==text,(row['protocol'],repr(binding.render(ids)),repr(text))
 assert count==len(ids)
 checked.append({'protocol':row['protocol'],'tokens':len(ids),'text_matches_selected_bytes':True,'final_native_ids_available':row['protocol']=='openai'})
state=json.loads((root/'exit-state.json').read_text());assert state['ExitCode']==0 and not state['OOMKilled']
result={'status':'pass','responses':checked,'all_journal_chains_valid':True,'native_final_token_ids_total':sum(x['tokens'] for x in checked if x['final_native_ids_available']),'shutdown_exit_code':0,'oom_killed':False,'binding_sha256':binding.digest,'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()},'scope':'Exact native final IDs for OpenAI; Anthropic text reconstructed from private selected tokens, not independently returned IDs'}
(public/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
