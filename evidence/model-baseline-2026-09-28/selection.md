# Model preflight

Qwen3.5-27B MLX 4-bit, revision 45797d2985a12c55e6473686e9ea91b95e959553, would require about 16.1 GB. Internal disk has about 10 GiB free. External Archive volume has ample space but the attempted dry-run local-dir setup failed with PermissionError: [Errno 1] Operation not permitted: /Volumes/Archive/keyprint-models. No model weights downloaded there; no permissions changed or user files deleted.

Qwen3.5-9B MLX 4-bit, revision 8b2b98c00a6b4d291155e4890773ca8f769aee53, is the feasible newer-family baseline. Choice made before generating any outputs; not selected by study success. Qwen3.5 support exists in installed mlx-lm 0.31.2. Its tokenizer and architecture differ from the pinned Qwen3-8B SDK; this baseline cannot be called a supported Keyprint integration. Use ordinary upstream inference first, retain all failures and unchanged source tasks. No private Qwen3-8B key or detector acceptance transfers.

Sources: https://huggingface.co/Qwen/Qwen3.5-9B and https://huggingface.co/mlx-community/Qwen3.5-9B-4bit . Family performance claims motivate an experiment; they do not prove factual fidelity here.
