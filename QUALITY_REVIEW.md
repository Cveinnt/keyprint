# Llama 3.2 3B local quality review

Current safeguard: SDK rewriting is blocked before inference. The observations
below describe historical runs. [Output quality contract](OUTPUT_QUALITY.md)
defines the required language, facts, roles, conditions and provenance checks.

Twenty fresh ordinary/marked outputs completed with no engineering failures.
They still contain invented details, factual errors and meaning drift. This
screen does not approve automatic rewriting or production output quality.

The model is `bartowski/Llama-3.2-3B-Instruct-GGUF` at revision
`5ab33fa94d1d04e903623ae72c95d1696f09f9e8`, file
`Llama-3.2-3B-Instruct-Q8_0.gguf`, SHA-256
`b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921`.
The September 20, 2026 run used llama-cpp-python 0.3.35 on macOS ARM64 CPU,
two threads, a 2,048-token context, temperature 0.7 and top-k 100. Model assets
retain their own Llama 3.2 license; they are not distributed with this SDK.

## What ran

- Six existing integration prompts, both conditions: 12 outputs, 603 tokens,
  zero mechanical flags. Actual local OpenAI and Anthropic HTTP requests also
  passed. No hosted provider inference occurred.
- Four longer cases declared before execution, both conditions: eight outputs,
  1,487 tokens, three outputs with mechanical flags. One case invokes the actual
  SDK `rewrite` method with protected source phrases.
- Two constructed provider-object rewrite probes per suite ran separately.
  The six-case suite's Anthropic-object rewrite duplicated text and failed
  lexical checks; the other three probes returned `needs_review`.

These are independent samples, not paired fixed-randomness quality estimates.
The review below is post-hoc and unblinded, written by the development assistant.
It is neither human approval nor evidence that watermarking caused any specific
error. Passing lexical screens does not establish meaning preservation.

## Review of every fresh pair

| Case | Ordinary output | Marked output |
| --- | --- | --- |
| Sky explanation | 40 words; relevant two-sentence scattering explanation. | 41 words; relevant two-sentence scattering explanation. |
| Meeting email | 55 words; retains the requested move from Tuesday 09:30 to Wednesday 14:00. | 53 words; retains times but adds an availability claim and repeats the original and proposed slots as alternatives. |
| Backup negation | Retains the prohibition until Maya confirms a successful restore. | Retains the same prerequisite. |
| Spanish leaf explanation | 48 words; includes the incorrect pigment term `xantófito` and an imprecise color explanation. | 48 words; incorrectly ties the change to the end of the longest night and dying cells releasing colors. |
| French review request | 33 words; retains deadline but mixes informal/formal address and misspells a document placeholder. | 39 words; retains deadline but invents prior collaboration on the document and uses awkward address. |
| JSON | Exact requested values and types, without a wrapper. | Exact requested values and types, without a wrapper. This unconstrained sample does not qualify a JSON mode for GGUF. |
| Long English project update | 145 words; retains approval cutoff, postponement and backup condition; adds that the rehearsal finished successfully, though the source only says finished. | 134 words; retains critical conditions but invents progress to a next stage and work already underway on issues. |
| Spanish approval email | 103 words; retains approval/delay/backup conditions, but presents the review deadline as a scheduled review at that time. | 88 words; retains the main conditions, while adding a comment-resolution requirement before finalization. |
| French approval email | 118 words; retains critical conditions; awkward deadline wording. | 154 words exceeds the 140-word cap; `09h30` and `14h00` retain the clock times but fail exact formatting. Adds urgency/progress language. |
| Long SDK rewrite | 161 words; changes the postponement trigger from missing approval to missing response. Retains the separate prohibition on starting without approval. Duplicates protected `Atlas`. | 176 words; retains main approval/postponement/backup conditions but duplicates protected `Atlas` and uses looser confirmation wording. |

The leaf explanations were checked against [Royal Botanic Gardens, Kew](https://www.kew.org/read-and-watch/why-do-leaves-change-colour): autumn changes in chlorophyll expose other pigments, with day length and night temperature contributing to the process. Neither a longest-night ending nor cells releasing colors is an adequate explanation.

Word counts above split on whitespace. The French time flags are formatting
differences, not evidence that the clock times changed. Both long rewrites
return `failed_checks` because protected phrase counts changed from one to two.
That conservative lexical guard remains intact, but it does not itself catch
the ordinary rewrite's change from approval to response. No scores or human
acceptances are inferred from this review.

## Decoder failure retained and fixed

Before these fresh runs, two French responses failed rendering. The native
full-sequence decoder removed the sampled space before `?`. Keyprint now
re-reads native token pieces with correctly sized buffers, preserves their
bytes and checks strict UTF-8. Exact replay of the original entropy and prompts
preserves all 595 Llama tokens, model calls, raw-head/probability hashes and
draws. A separate replay preserves all 1,002 preceding SmolLM2 tokens. Original
failures remain in the private development receipts. Replays are not counted
as fresh quality samples. Sampling arithmetic is unchanged.

## Reproduce and inspect

Use the [pinned download recipe](README.md#local-gguf-with-llamacpp) and an
installed wheel with `llama-cpp`, `clients` and `server` extras. From the checkout:

```sh
python tools/validate_compatibility.py --backend llama-cpp \
  --model models/llama3.2/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --output llama-integration --http-client
python tools/validate_compatibility.py --backend llama-cpp \
  --model models/llama3.2/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --cases tools/quality_cases.json --output llama-longform
```

Each output directory contains `public/comparison.html` and JSON with every
attempt, full prompts, both texts, runtime identity and review flags. Root run
directories also contain private keys and journals; do not publish those.
New runs use new randomness, so outputs will differ. The six-case word-limit
screens were added to match the prompts' existing obligations after the initial
decoder run, before fresh generation. They do not retroactively change old
reports. No thresholds were relaxed to pass this screen.
