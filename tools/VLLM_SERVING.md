# Native vLLM CPU HTTP pilot

This is a pinned reproduction environment, not a production server recipe.
It uses the server's actual sampling hook. No Keyprint wrapper server sits
between the client and vLLM. The final local run used two CPU threads, a 4 GiB
container limit, 256 MiB KV cache, float32 and the recorded SmolLM2 assets.

Create a fresh private receipt directory with `owner.key` and `api.key`, each
32 random bytes, mode 600. Keep the directory mode 700 and create a mode-700
`traces/` subdirectory. Mount it at `/results`, the pinned model directory at
`/model` (read-only), this checkout's `src/` at `/keyprint/src` (read-only),
and `tools/` at `/keyprint/tools` (read-only).

Use image:

```
vllm/vllm-openai-cpu@sha256:527ec4e8188f2ad480aca5863ab3b7e7c39cfda84f6c0bbb06525363a3eb5a0f
```

Set `PYTHONPATH=/keyprint/src`, `KEYPRINT_KEY_FILE=/results/owner.key`,
`KEYPRINT_TRACE_DIR=/results/traces`, `VLLM_CPU_OMP_THREADS_BIND=0-1`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`.
Use `--cpus 2 --memory 4g --shm-size 1g` and publish only
`127.0.0.1:8832:8000`. Start `/opt/venv/bin/python
/keyprint/tools/serve_vllm_pilot.py`. The entrypoint reads the local API token,
registers the fixed processor class server-side and leaves model generation
configuration explicit. It never accepts serialized processor code from a client.

Once `/v1/models` succeeds with the local bearer token, run from the host with
the OpenAI and Anthropic clients installed:

```sh
python tools/validate_vllm_server.py --root private-vllm-run --attempt attempt-1
```

Four requests run concurrently; two more use Anthropic's client. The test
rejects incompatible temperature before sampling, disconnects a stream after a
received native token, and checks recovery. It does not test same-key retries:
native vLLM does not inherit Keyprint preview-server idempotency. Disable client
retries. Anthropic 1.6.0 needs local sampling overrides via `extra_body` and
an explicit bearer header for this vLLM endpoint; the script shows both.

Stop the container gracefully and save its `.State` JSON as `exit-state.json`
in the private root. Then independently reconcile receipts:

```sh
python tools/audit_vllm_server.py --root private-vllm-run \
  --assets /path/to/pinned-smollm2 --attempt attempt-1
```

Only `public/` is shareable. Source/model hashes, journals, complete outputs,
failed attempts and shutdown state must be retained. The September 21 result
matched 99 returned OpenAI token IDs, plus 50 Anthropic selected-token bytes and
counts. Anthropic did not independently return token IDs. Native disconnect
stopped after two selections, before its cap of 384. Exit code was 0, no OOM.
This is one bounded CPU lifecycle pilot; GPU, sustained load, arbitrary model
families, exhaustive streaming equivalence and calibrated detection remain open.
