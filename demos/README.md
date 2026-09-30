# A starting point, not a screenshot

[Open the gallery](https://keyprint.vercel.app/play/) · [Download the viewer](https://keyprint.vercel.app/play/keyprint-demo.zip) · [Share your remix](https://github.com/Cveinnt/keyprint/discussions/categories/show-and-tell)

Three retained real Qwen/MLX runs: an explanation, an email and French prose.
Choose a run, highlight wording differences, replay tokens or scrub measured
prefixes. The email's invented time/location is a disclosed failure, not a polished
success story. No inference, account, API key or model download during playback.

## Run the demos with one command

From this cloned repository, Python 3.12 or 3.13:

```sh
python tools/serve_demos.py
```

Open the printed localhost URL. This uses only Python's standard library, binds
to 127.0.0.1, and removes its temporary build when stopped. `--port 8080` changes
the port. To build files for your own static hosting instead:

```sh
python tools/serve_demos.py --output my-demo
```

Use a new output directory. Serve the downloaded ZIP with `python -m http.server`
after extracting it; browser `file://` URLs cannot reliably fetch its JSON.

## Make something different

- **Token timeline:** `experiment.outputs.marked.trace` contains exact token IDs,
  bytes and text offsets. Render the committed tokens; do not invent alternative probabilities.
- **Signal sketch:** `experiment.outputs.marked.inspection.series` contains
  independently retokenized prefixes. Plot `matching` and `control` against
  `characters`. Null stays unavailable; these are not authorship probabilities.
- **Paired reading view:** `experiment.outputs.ordinary.text` and `.marked.text`
  are the original generated responses. Display both and retain identical pairs.
- **Your own gallery:** generate comparisons with the [Python recipes](../examples/README.md),
  then call `export_gallery(...)`. No extra inference occurs during export.

Start with [replay.json](recordings/replay.json) for one comparison or
[gallery.json](recordings/gallery.json) for all three. Those public recordings
are unchanged. [Provenance](recordings/provenance.json) records their historical
creation state; “private development” describes when they were generated.

The viewer is built from the SDK's shared HTML, CSS and JavaScript by
[`serve_demos.py`](../tools/serve_demos.py). Change presentation without adding
a framework to the core library. Deep links use `?example=0`, `1` or `2`, followed
by `#outputs`, `#choices` or `#explore`.

Prompt editing in recorded mode prepares runnable Python. It never pretends to
produce a fresh model response. To run new prompts, use the [local playground](../README.md#start-here).
Review your exported prompt/output before posting; keys and private journals stay private.
