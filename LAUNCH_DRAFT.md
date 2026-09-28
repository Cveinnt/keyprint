# Launch copy, held for release verification

Draft only. Nothing here is scheduled or posted. The GitHub repo is private and
the new `keyprint` package is not publicly installable. Replace placeholder links
only after anonymous access and installation have been verified.

The launch should earn attention through a useful, inspectable watermarking
tool. Pretext is a benchmark for impact and presentation, not a feature or size
checklist. No copy or demo may trade semantic fidelity or clue conformance for
speed, smaller assets, stronger-looking signals or a more impressive example.

## X opening

I built Keyprint to make text watermarking something you can actually play with.

Generate two responses. Compare the wording. Replay the model's choices. Edit
the text and inspect the pattern underneath.

A Python SDK, a local playground, and demos you can remix.

[Verified demo link]

## Follow-up

The mark is applied while a supported local model generates. No translating or
rewriting its response afterward.

The same comparison object powers the demo and your own visualizations. You can
export a recorded gallery without adding a frontend framework.

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
comparison and a 48-output candidate experiment. We rejected the candidate:
more varied answers did not compensate for losing detection signal.

I'd like feedback on the API and the experiments people want to build with it.

## Evidence and publishing conditions

- Link the actual demo, public repository, installed package and scope table.
- Keep the real prefilled run and custom-prompt path immediately accessible.
- Do not pitch wording highlights as watermarked-word attribution, diagnostic
  fractions as confidence, or repeated marked outputs as unchanged diversity.
- Do not claim 25/25 conformance, exact reverse engineering, universal native
  framework support or semantic preservation for every generated response.
- Keep the original release hold until product verification is complete. No
  registry or social scheduling action is implied by preparing this copy.
