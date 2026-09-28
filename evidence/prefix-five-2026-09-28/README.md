# Five layers did not solve the fidelity/detection tradeoff

**Rejected for promotion.** This separately identified research candidate runs
the first five tournament layers instead of all thirty. It changes probabilities
during native generation, never rewrites or translates a completed response.
The SDK and its default sampler remain unchanged.

All 48 planned generations completed at EOS without inference/scoring errors.
All **5,211 committed tokens** reconcile with native bytes and journal records;
all 48 random-draw transcripts are distinct. All outputs remain in their requested
language. Those execution checks do not establish factual or semantic fidelity.

| Frozen development measure | Ordinary | Current reference | Five layers |
| --- | ---: | ---: | ---: |
| Task rubric passes | 12/16 | 14/16 | 11/16 |
| French explanation passes | 5/8 | 6/8 | 5/8 |
| Approval email passes | 4/4 | 4/4 | 4/4 |
| Longer English passes | 2/2 | 2/2 | 2/2 |
| Longer Spanish passes | 1/2 | 2/2 | 0/2 |
| Longer matching-key detections under registered rule | 0/4 | 3/4 | 1/4 |
| Longer other-key detections under registered rule | 0/4 | 1/4 | 0/4 |

The five-layer candidate loses both task passes and detection hits in this small
screen. It is not justified as a quality fix. Its paired outcomes against the
reference are ten both-pass, four reference-only, one candidate-only and one
both-fail. More variation does not suffice: all four candidate French pairs differ,
while one reference pair repeats exactly. Two samples per key cannot establish
a general diversity rate.

The candidate's Spanish failures include omission of the required return date
and a 215-word answer exceeding the explicit limit. A French answer incorrectly
ends with an impression of a blue night. The current reference retains both
known upward-reflection failures under key 2. Ordinary generation also fails.
This does not establish the watermark as the cause of every task failure.

Four French ratings conservatively fail ambiguous wording. They were flagged
before revealing conditions. Counting all four as passes gives **14/16 ordinary,
14/16 reference and 13/16 candidate**. Detection results do not change, so the
candidate remains rejected. Primary ratings are preserved; see
[sensitivity.json](sensitivity.json).

## Detection is still unqualified

The reference uses the unchanged thirty-layer weighted rule. The candidate uses
an exact fair-binomial tail for the first five layers, with the same nominal
0.005 per-key cutoff and two-key family target. These are comparative development
rules, not verified deployment error rates.

Both detectors were applied to every output. The five-layer score detects only
1/4 longer reference outputs as well as 1/4 candidate outputs. The thirty-layer
score detects 3/4 reference and 1/4 candidate outputs. Neither detector flags any
of the sixteen ordinary outputs under either tested key. The thirty-layer rule
does flag one reference output under the wrong key. That event remains visible;
it prevents treating these hits as reliable owner-attribution evidence.

The candidate's twelve short outputs have one matching hit; the reference's
twelve have none. Neither supports a short-text detection claim.

An additional five-layer diagnostic on **500 previously opened unwatermarked
responses** flags four under either of two existing keys. Its IID-only 97.5%
upper bound is **2.04%**, not proof of a 1% deployment false-positive rate.
The data were previously examined, and a response-level bound does not resolve
fixed-key dependence or domain transfer. See [null summary](opened-null/summary.json).

The null source is [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC BY-SA 3.0.
Only source indices, hashes and derived measurements are exported, not source
responses. This diagnostic reuses the prior 500-response selection and keys.

## Evidence and decision

- [All 48 samples](REVIEW_SAMPLES.md), including failures and exact repetitions.
- [Plan](plan.json), [method and reproduction](METHOD.md), [frozen ratings](frozen-ratings.json),
  [rating commitment](rating-commitment.json), [results and receipt audit](results.json).
- 49 targeted local tests pass. All 39 frozen engine files pass integrity checks.
  Registered script and candidate hashes match the files used for inference.
- No scientific clue closes. No SDK default, dependency, website design, hosted
  CI setting, publication or release threshold changes.

Two simple attenuation candidates have now failed their registered screens.
Stop treating reduced concentration alone as the remedy. The next qualification
must separate base-model factual limitations from watermark effects on broader,
fresh tasks, while testing detector sensitivity and wrong-key behavior together.
Keep the stronger reference as the comparison baseline; neither failed candidate
belongs in the product merely to produce more varied demos.
