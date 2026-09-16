"""Trusted offline pilot in the pinned SGLang ARM image; no public server."""
import json
from pathlib import Path

import sglang as sgl
from transformers import AutoTokenizer
from keyprint.experimental.sglang import KeyprintLogitsProcessor


def main():
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True, trust_remote_code=False)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                              add_generation_prompt=True)
               for text in ("Explain why the sky is blue in two sentences.", "Write a short thank-you email.")]
    llm = sgl.Engine(model_path="/model", device="cpu", dtype="float32", tp_size=1,
        max_running_requests=2, context_length=512, max_total_tokens=1024,
        disable_overlap_schedule=True, disable_radix_cache=True, disable_cuda_graph=True,
        enable_custom_logit_processor=True, mem_fraction_static=.5)
    try:
        params = {"max_new_tokens": 64, "temperature": 1., "top_p": 1., "top_k": -1,
                  "custom_params": {"keyprint_condition": "marked"}}
        outputs = llm.generate(prompts, params, custom_logit_processor=KeyprintLogitsProcessor.to_str())
        Path("/results/outputs.json").write_text(json.dumps(outputs, indent=2, default=str))
        print(json.dumps(outputs, default=str), flush=True)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
