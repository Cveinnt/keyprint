"""Run inside the pinned CPU image with public model assets and private traces."""
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from keyprint.experimental.vllm import KeyprintLogitsProcessor


def main():
    llm = LLM(model="/model", dtype="float32", max_model_len=512,
              enforce_eager=True, max_num_seqs=2, max_num_batched_tokens=512,
              kv_cache_memory_bytes=256 * 1024 * 1024, enable_prefix_caching=False,
              logits_processors=[KeyprintLogitsProcessor])
    params = SamplingParams(max_tokens=64, temperature=1., top_p=1., top_k=-1,
                            extra_args={"keyprint_condition": "marked"})
    responses = llm.chat([[{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                          [{"role": "user", "content": "Write a short thank-you email."}]],
                         params, use_tqdm=False)
    results = [{"request_id": r.request_id, "text": r.outputs[0].text,
                "token_ids": list(r.outputs[0].token_ids), "finish_reason": r.outputs[0].finish_reason}
               for r in responses]
    Path("/results/outputs.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
