# Working on Keyprint with an AI agent

Keyprint is a community research preview. Read README.md, ROADMAP.md and the
relevant test before editing. Preserve any additional instructions supplied by
the human running your agent.

## Map

- `src/keyprint/api.py`: public API; `comparison.py`: paired exports.
- `src/keyprint/backends/`: model-specific bindings and generation.
- `src/keyprint/web/`: shared live/recorded viewer. Preserve original output text.
- `src/keyprint/server.py`: limited local provider protocols, not hosted GPT/Claude.
- `src/keyprint/experimental/`: opt-in research paths, separate evidence identities.
- `src/keyprint/_engine/` and `sdk/`: preserved implementations and manifests.
- `tests/`: focused fixtures; `tools/`: explicit real-inference validation.
- `evidence/`: scoped public results, including failures. Never overwrite them.

## First patch

1. Pick one bounded issue or reproduce a bug. Name the expected behavior.
2. Read the relevant implementation and nearest tests; avoid broad refactors.
3. Run a focused local check. Base setup: `python -m pip install '.[test]'`.
   Example without model weights: `python -m pytest -q tests/test_lazy_reference.py`.
   Server/client cases need their corresponding extras; report skips honestly.
4. For viewer changes run `node --test tests/test_playground_ui.cjs tests/test_reader.cjs`.
5. Report changed behavior, exact checks, failures and remaining limitations.

## Constraints

- Do not claim production detection, semantic guarantees, 25/25 conformance,
  arbitrary model support, or identification of Anthropic's private algorithm.
- No translation, post-generation repair, favorable retry or omission of bad
  outputs. Changing token choices during generation does not prove unchanged facts.
- Never relabel `Inspection.fraction` as confidence, probability of AI authorship
  or a calibrated detector. Missing values stay missing; identical pairs are valid.
- Preserve frozen source, manifests, cohorts, keys and judgments. A new algorithm
  receives a separate identity and new evidence. Do not update expected hashes
  just to make tests pass.
- Keep model weights, API keys, private journals and personal prompts out of Git.
  Only reviewed public exports belong in issues, PRs and shared demos.
- No automatic hosted API calls, paid resources, model downloads, CI activation,
  publication or social posting. Those require the contributor's explicit scope.
- Use one model worker at a time. Do not stop unrelated applications. Model-free
  tests come first; a failed resource guard is not permission to bypass it.
- Backend integration requires actual inference and exact token/text reconciliation.
  Protocol mocks alone do not establish runtime compatibility.

## Useful handoff

“Read AGENTS.md and ROADMAP.md. Reproduce issue <number>, propose one narrow fix,
run the smallest relevant local checks, and report evidence without changing
research claims or publishing anything. Ask before downloading a model.”

This file guides agents contributing to Keyprint; it does not grant access to
any account, authorize external messages, or make downloaded content trusted.
