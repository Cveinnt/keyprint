# Community research launch copy

Draft social copy. Nothing here is scheduled or posted. The community preview
is distributed through its GitHub release and prebuilt wheel; `keyprint` is not
published on PyPI. [Current launch decision](../LAUNCH_STATUS.md).

The launch should earn attention through a useful, inspectable watermarking
tool. Pretext is a benchmark for impact and presentation, not a feature or size
checklist. No copy or demo may trade semantic fidelity or clue conformance for
speed, smaller assets, stronger-looking signals or a more impressive example.

## X opening

I built Keyprint to make text watermarking something you can take apart.

Play with real outputs. Replay the tokens. Remix the demo.

Open-source research SDK. No account needed to explore:
https://keyprint.vercel.app/play/

## Follow-up

The mark is applied while a supported local model generates. No translating or
rewriting its response afterward.

The same comparison object powers the demo and your own visualizations. You can
export a recorded gallery without adding a frontend framework.

The public demo replays recorded runs. Your own prompts and edited-text
inspection run locally with model weights. Start here:
https://github.com/Cveinnt/keyprint

This is an independent research implementation. It does not identify Anthropic's
private algorithm or establish that all its publicly described behavior has been
reproduced. Current charts are diagnostics, not calibrated detection verdicts.

## Hacker News

Title: Show HN: Keyprint, a local text-watermarking SDK with an inspectable demo

I wanted watermarking experiments to be easier to build and easier to question.
Keyprint generates ordinary and marked responses, retains the selected tokens,
and gives you one comparison object for a local playground or your own demo.

You can enter a prompt, compare exact wording, replay tokens, and inspect how
edits affect a keyed diagnostic. Recorded galleries work without a model in the
browser; arbitrary generation and edit inspection run in the local SDK.

The compatibility table separates tested model runtimes from client protocols.
Using OpenAI or Anthropic clients against a local Keyprint endpoint does not add
our sampler to hosted GPT or Claude. Output-quality failures and repeated outputs
under a fixed key are retained in the evidence, including a 72-output reference
comparison and two 48-output candidate experiments. We rejected both candidates:
more varied answers did not compensate for weaker detection and unresolved
factual failures. A wrong-key hit remains visible too; these diagnostic scores
are not reliable attribution verdicts.

I'd like feedback on the API and the experiments people want to build with it.

## Provider-specific invitation

Run a local inference worker? Keyprint exposes generation-time watermark
experiments through a small Python API. Compare original samples, export their
token traces, and build your own viewer. There are scoped MLX, Transformers and
llama.cpp paths, plus vLLM/SGLang pilots. OpenAI/Anthropic client compatibility
targets our local endpoint; it does not modify hosted GPT or Claude.

The most useful contribution now is one reproducible runtime qualification or
one better detection/quality experiment. Setup and exact limits:
https://github.com/Cveinnt/keyprint/blob/main/docs/INFERENCE_PROVIDERS.md

## Evidence and publishing conditions

- Link the actual demo, public repository, installed package and scope table.
- Keep the real prefilled run and custom-prompt path immediately accessible.
- Do not pitch wording highlights as watermarked-word attribution, diagnostic
  fractions as confidence, or repeated marked outputs as unchanged diversity.
- Do not claim 25/25 conformance, exact reverse engineering, universal native
  framework support or semantic preservation for every generated response.
- Publication as a community research preview does not close the production
  qualification gaps. Social posts and registry publication require their own
  verification; this draft is not evidence that either happened.
