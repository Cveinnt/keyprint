# Build your own Keyprint experiment

Private development SDK. Start with the root README's local install. These
recipes use real local inference, not the browser teaching illustration.

## Prefilled interactive gallery

`python examples/gallery.py /path/to/qwen3-8b-4bit my-gallery`

Generate explanation, email and French pairs, then serve with
`python -m http.server --directory my-gallery`. Click an example to switch all
of its real data: prompt, both responses, exact token trace, prefix curve and
remix recipe. Edit `PROMPTS` to make your own collection. The public helper
`export_gallery({"Title": comparison}, "new-directory", notes={"Title": "Review note"})`
exports existing comparisons without more model calls. Review the text before
sharing; generation can still invent details or fail instructions. The retained
development email example violates its no-time/location requirement and is
explicitly flagged. No sample is silently repaired or regenerated for display.

## 1. Share a paired response

`python examples/compare_live.py /path/to/qwen3-8b-4bit my-demo`

Edit the prompt, generate once, export the same viewer as the playground. Open
with `python -m http.server --directory my-demo`. Playback is immediate and
labeled recorded; prompt edits lead to matching Python code for a live run.

## 2. Explore how the signal changes with length

`python examples/prefix_signal.py my-demo/replay.json`

A dependency-free reader of the actual exported measurements. Feed these rows
to Canvas, SVG or your own renderer. Null values stay unavailable. Prefixes use
Unicode code points and are independently retokenized; they are not per-word
watermark attribution. No detector threshold or confidence is invented.

## 3. Compare two supported local models

`python examples/compare_models.py /path/to/smollm2 /path/to/smollm3 gallery`

Loads models sequentially through the Transformers backend, generates ordinary
and marked responses to the same prompt, and exports one viewer per model. This
is an exploratory comparison, not a quality or speed benchmark. Each model has
its own scoped qualification requirements. Use a new output directory.

All recipes retain original output, including truncation and identical pairs.
Review prompts and generated text before sharing the exported folders. Private
journals and keys stay outside the exports. No data is uploaded.
