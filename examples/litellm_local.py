"""Optional LiteLLM route to a local Keyprint server; no core dependency.

Install litellm==1.102.0 separately. Start the server as in PROVIDERS.md.
"""
import os
from pathlib import Path
from uuid import uuid4

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import litellm

litellm.telemetry = False
router = litellm.Router(
    model_list=[{
        "model_name": "watermarked-local",
        "litellm_params": {
            "model": "openai/keyprint",
            "api_base": "http://127.0.0.1:8765/v1",
            "api_key": Path("local-api.key").read_bytes().hex(),
            "max_retries": 0,
        },
    }],
    num_retries=0,
    max_fallbacks=0,
    fallbacks=[],
    cache_responses=False,
    timeout=180,
)
request_id = str(uuid4())
try:
    response = router.completion(
        model="watermarked-local",
        messages=[{"role": "user", "content": "Explain why a seed needs water."}],
        max_tokens=96,
        extra_headers={"Idempotency-Key": request_id},
    )
    print(response.choices[0].message.content)
finally:
    router.reset()
# Keep the same route, ID and body to recover an accepted timed-out attempt.
# Do not configure hosted/unmarked fallbacks for a watermarked route.
