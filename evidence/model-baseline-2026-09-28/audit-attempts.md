# Retained audit corrections

The first independent audit used text_config.eos_token_id (248044), which does not match the generated termination token, and failed with `ValueError: EOS does not match the terminal event`. All eight generation receipts had independently passed reconciliation against the actual MLX tokenizer at generation time.

Inspection of installed mlx_lm.utils.load shows it passes only top-level config.eos_token_id to load_tokenizer. That field is absent; tokenizer_config names `<|im_end|>`, resolving to 248046. All eight outputs end at 248046. The nested config instead records 248044. Auditor now reconstructs the actual upstream loader policy and records both values explicitly. No prompt, output, generation script, seed, rating or model artifact changed; no inference rerun. This mismatch is a material native-binding concern, not evidence of compatibility.

An inspection attempt initially imported load_tokenizer from mlx_lm.tokenizer_utils and failed with ImportError. Read installed load() source and used mlx_lm.utils.load_tokenizer, its actual imported binding.
