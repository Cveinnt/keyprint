# Third-party material in keyprint-research-v3 0.0.4rc3

Project code: MIT, Copyright (c) 2026 Vincent Wu (Cveinnt). Bundled third-party assets retain their own licenses and attribution.

## Included Qwen assets

The following files are copied without byte changes from
[`mlx-community/Qwen3-8B-4bit`, revision `545dc4251c05440727734bcd94334791f6ab0192`](https://huggingface.co/mlx-community/Qwen3-8B-4bit/tree/545dc4251c05440727734bcd94334791f6ab0192):

| Wheel member | SHA-256 |
|---|---|
| `keyprint_v3/_bundle/release/keyprint-0.0.3rc1/installed-audit/keyprint/_impl/release/tokenizer/qwen3-8b-4bit/tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `keyprint_v3/_bundle/release/keyprint-0.0.3rc1/installed-audit/keyprint/_impl/release/tokenizer/qwen3-8b-4bit/config.json` | `e5485285fd7e289e76e9cffa112f6dc2e3426519082f7db9b69041589f81a218` |

The pinned MLX model card identifies Qwen/Qwen3-8B as its source, describes the
conversion using mlx-lm 0.24.0, and declares Apache License 2.0. No model weights
are included in this SDK. The configuration is the converted MLX configuration;
it is not asserted byte-identical to the original non-MLX Qwen configuration.

Upstream attribution: **Copyright 2024 Alibaba Cloud**, as retained in the
[Qwen license file](licenses/Qwen-Apache-2.0.txt). Its source is the
[Qwen/Qwen3-8B license pinned at `b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/LICENSE).
The MLX card originally links the moving `main` license; this review records the
resolved revision. That revision is license evidence, not a claim about the
base-model revision used during MLX conversion.

No file named NOTICE appeared in the captured MLX root listing or Qwen file
inventory. This observation is limited to those recorded upstream inventories.

## Research reference

SynthID-Text is a public mathematical and experimental reference for this work:
[`google-deepmind/synthid-text`, revision `addb4a158143c7c6851a1308f78b89fceed59683`](https://github.com/google-deepmind/synthid-text/tree/addb4a158143c7c6851a1308f78b89fceed59683).
Its upstream [Apache-2.0 license](licenses/SynthID-Text-reference-Apache-2.0.txt)
is retained here as reference evidence. The inspected SDK wheel contains no
`synthid_text` package or import. The historical nested v1 SDK inventory describes 18 implementation modules from
the local research files recorded by its provenance manifest. The current v3
distribution additionally contains the frozen v2/v3 adapters, support filter,
caller and reporting modules, plus the public package wrapper. The bundled
`keyprint_v3/bundle-manifest.json` inventories the frozen bundle; it does not
inventory the outer wrapper. Historical inventory statements are preserved
inside that bundle and do not describe the entire current wheel. These inventories
do not certify independent authorship or establish a blanket license for local files.

## Separately installed dependencies

The wheel declares `numpy==2.5.2`, `scipy==1.17.1` and `tokenizers==0.23.1`,
and requires Python >=3.12. NumPy determines this minimum Python version.
Their packages are not vendored inside this wheel. Their own distributions carry
their applicable notices. A future offline installer or bundled environment would
need a separate inventory of the exact dependency distributions it includes.

## Project code

The MIT grant applies to project-authored code and documentation. The bundled Qwen tokenizer/configuration remain under their upstream Apache-2.0 terms. SynthID-Text is a research reference, not an endorsement or an included detector. No model weights, private study keys or third-party evaluation corpora are redistributed.
