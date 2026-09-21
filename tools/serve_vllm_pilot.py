"""Pinned CPU container entrypoint. Requires /model and private /results keys."""
from pathlib import Path
import os,sys
args=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model','--served-model-name','keyprint','--host','0.0.0.0','--port','8000','--dtype','float32','--max-model-len','512','--enforce-eager','--max-num-seqs','2','--max-num-batched-tokens','512','--kv-cache-memory-bytes','268435456','--no-enable-prefix-caching','--generation-config','vllm','--logits-processors','keyprint.experimental.vllm:KeyprintLogitsProcessor','--api-key',Path('/results/api.key').read_bytes().hex()]
os.execv(sys.executable,args)
