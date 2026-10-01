# Launch decision · October 1, 2026

**Ready to share as an open research toolkit. Not ready to sell as production
watermarking, universal inference integration, or a verified reconstruction of
Anthropic's private algorithm.** Publication does not close the research gates.

## Against the Pretext launch bar

[Pretext's public README](https://github.com/chenglou/pretext) combines a registry
install, a small API, live demonstrations and reusable examples. Those are the
relevant product qualities; copying its architecture or size is not the goal.

| Dimension | Keyprint today | Remaining difference |
| --- | --- | --- |
| Immediate demonstration | Public, prefilled, account-free gallery; exact wording, token replay and prefix charts | Real new generations need a local model; browser gallery is recorded |
| Creative reuse | Export a comparison/gallery, download the viewer, build from `to_dict()` | No independent community showcase or adoption established |
| Installation | Tagged GitHub release and verified prebuilt wheel; `keyprint` import and CLI; optional backend extras | No `pip install keyprint` registry release yet; model weights and RAM are additional |
| Presentation | Serif-led site, illustrated README, share links and social-preview metadata | Three featured recordings are a starting collection, not a mature demo ecosystem |
| Provider integration | Scoped actual inference on MLX, Transformers and llama.cpp; client recipes | vLLM/SGLang remain pilots; no native Ollama/LM Studio hook, hosted GPT/Claude sampling hook or broad production qualification |
| Core promise | Inspectable, reproducible generation-time watermark experiments | Reliable detection and negligible output-quality impact are not established |

The largest difference is the last row: presentation cannot substitute for a
dependable underlying capability. Do not describe the project as Pretext-level
production readiness or promise equivalent adoption.

## Exact clue status

The [public historical ledger](https://keyprint.vercel.app/clue-ledger.json)
records **22/25 scoped reference acceptances**, with three open:

- **A02, output quality:** the ledger's conservative harm bound is 7.50%, above
  its 5% target. Newer candidate failures remain in [ROADMAP.md](../ROADMAP.md).
- **A03, reader indistinguishability:** semantic review is different from a
  blinded test of whether readers can identify marked output.
- **A18, serving overhead:** only 4/10 contrasts meet both ledger latency gates.
  These are project acceptance criteria, not a numeric SLA disclosed by Anthropic.

That historical count does not qualify every new SDK model, adapter or candidate.
Separately, the latest completed paced candidate detected 1/64 marked outputs at
its declared cutoff, with 0/64 ordinary hits. That does not establish useful
detection. Its strict task passes were 16/64 marked and 25/64 ordinary; neither
arm supplies a general quality guarantee. The balanced follow-up is incomplete.

## Finite release scope

Share the toolkit, the real demo and the invitation to reproduce or improve it.
Use [reviewed launch copy](maintainers/LAUNCH_DRAFT.md). Keep production claims on
hold until all three independent workstreams have evidence:

1. Fixed detector, declared false-positive criterion and fresh held-out power.
2. Paired language/fact/task evaluation plus the separate reader test.
3. A named runtime's serving lifecycle, load and overhead qualification.

These are bounded research/engineering projects, not reasons to keep redesigning
the landing page. Public contribution issues and `AGENTS.md` provide the handoff.
Social drafts are not scheduled posts; release assets are not a PyPI publication.

## October 1 distribution verification

The release wheel was built from unchanged tag `v0.1.0a1`, commit
`07a517faca8f960b716f928c438ed371dadc7246`. All 92 packaged source/assets match
that checkout byte for byte. A fresh Python 3.13 environment passes
`keyprint doctor` and 87 focused installed-wheel tests; one provider-object test
skips because optional OpenAI/Anthropic clients are absent. No inference was run.
The wheel is 2,374,541 bytes, excluding its dependencies and model weights.
The release's `SHA256SUMS` and `BUILD_PROVENANCE.json` identify the exact artifact.
