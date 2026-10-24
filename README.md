# Paraphrase Check

A `nanoagent.plugin/v0001` plugin. A model reads four pairs of Wikipedia
sentences that share almost all their words and decides, for each pair, whether
the two sentences mean the same thing. A deterministic verifier grades the
verdicts against human labels, which the model never sees.

## Tools

| Tool | Does | Code |
|---|---|---|
| `read_pairs` | Shows a batch's 4 sentence pairs and, for each, whether its two sentences use exactly the same words and where they differ, word by word | `src/plugin/pairs.py` |
| `pin_pairs` | Reads the batch's answer key from the bundled snapshot, after checking its pinned SHA-256; the task's reference, run out of the model's sight | `src/plugin/pairs.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `judge-pairs` | Decides, for each of a batch's 4 pairs, whether its sentences mean the same or not | `decide`, with `read_pairs`; reference `pin_pairs` |

The input is a batch number from 0 to 224. The output is the batch number and
one verdict per pair, `p1` to `p4`, each `same` or `different`, each in a level
of its own (`first`, `second`, `third`, `fourth`). With the `decide` runner the
model writes no value of its own: it chooses the next step and the four
verdicts, and NanoAgent copies the batch number, the only integer in
`read_pairs`'s output. The decide runner asks each level in a request of its
own, so a whole task takes six model requests (all six RUSH Lab allows: two to
choose the next step, then one per pair). Asked two pairs at a time, the
deployed model, whose thinking counts toward its 2,048-token reply, often ran
out while going back and forth on one hard pair; the instructions therefore
also give a fixed three-step way to judge a pair.

What code can compute, the tool computes: `changes` lines the two sentences up
word by word (Python's `difflib` on the words split at spaces) and lists every
stretch that differs, and `same_words` says whether the sentences are the same
words in another order. What is left to the model is the one thing those cannot
tell: whether the move changed who did what.

### Data

The pairs come from **PAWS** (Paraphrase Adversaries from Word Scrambling;
Zhang, Baldridge and He, NAACL 2019; `https://github.com/google-research-datasets/paws`,
Google LLC), the test split of **PAWS-Wiki Labeled (Final)**, read from the
Hugging Face mirror `google-research-datasets/paws`, configuration
`labeled_final`. Each PAWS pair was made from a Wikipedia sentence by swapping
or moving its words, or by translating it into another language and back, and
human raters labeled whether the two sentences still mean the same. PAWS may be
freely used for any purpose; its license notice ships as
`src/plugin/PAWS-LICENSE.txt`.

The plugin bundles a selection, `src/plugin/snapshot.json.gz` (70 KB): 900
pairs, 450 of each kind, in 225 batches of 4, each pair with its PAWS id, its
two sentences as PAWS writes them (words and punctuation separated by spaces)
and its label. The snapshot's SHA-256 is pinned
(`58ee9520dbaf08a594ecb3f7111dba92d62675d7baab0f05b6f71960ac21aea5`), so a
changed file stops the plugin instead of silently changing the task.

On the whole test split, surface measures still beat a coin: the best
threshold on word-order similarity (`difflib` ratio over the words) tells the
labels apart with balanced accuracy 0.71, and on bigram overlap with 0.66,
because swaps that change the meaning tend to keep more of the order. So the
selection is **matched**: every bin of (word-order similarity, bigram overlap,
word overlap), in tenths of each, holds as many same-meaning pairs as
different-meaning ones. On the bundled pairs the best threshold rule then
reaches only:

| Measure | Whole test split | Bundled pairs |
|---|---|---|
| Word overlap (Jaccard) | 0.525 | 0.523 |
| Bigram overlap (Jaccard) | 0.656 | 0.520 |
| Word-order similarity | 0.706 | 0.533 |
| Length gap | 0.517 | 0.532 |
| Same words, any order | 0.532 | 0.514 |

`tests/test_snapshot.py` checks that each of these, and the number of
changes, stays below 0.56 on the bundled pairs. Pairs with the same sentence
twice or a sentence over 300 characters are left out, and every sentence
appears at most once in the snapshot. Each batch holds 1 to 3 pairs that mean
the same (a third of the batches each 1, 2 and 3, spread by SHA-256); the order
of the pairs, and which sentence is shown as A, also come from SHA-256 of the
pair ids. The build script is kept outside the plugin.

### What the verifier checks

`verifier/judging.py` checks, in order; any failure among the first five scores 0:

| Id | Constraint |
|---|---|
| `key-batch` | The answer key is for the task's batch |
| `key-source` | The answer key was read from the pinned snapshot |
| `key-pairs` | The answer key has 4 distinct pairs, p1 to p4, in order |
| `key-composition` | The batch has 1 to 3 pairs that mean the same, so both kinds are in it |
| `output-batch` | The output answers the task's batch |
| `p1-right` … `p4-right` | Pair p*n* is judged same when its sentences mean the same, and different when not |

The verdict is valid only when all four pairs are right. The score is
chance-corrected accuracy, max(0, (accuracy − *c*) / (1 − *c*)), where *c* is
the accuracy of the best constant answer for the batch: 1/2 for a batch with 2
pairs of each kind, 3/4 for one with 1 and 3. So on a 2-2 batch 4 and 3 right
score 1 and 0.5, and on a 1-3 batch only 4 right scores. Answering `same`
everywhere, `different` everywhere, or every verdict turned around scores 0; a
fair coin scores 0.104 in expectation (all 16 answers counted in
`tests/test_baselines.py`). Plain accuracy would give a constant answer 1/2 on
average and the better constant 2/3.

The tool checks (`verifier/pairs.py`, `verifier/pin.py`) refuse a batch number
out of range, a shown batch that is not p1 to p4 in order, a sentence that is
empty, padded or over 300 characters, a pair showing one sentence twice, a
`same_words` or `changes` that does not match the two sentences, and an answer
key that breaks any of the key constraints above.

### Limits

- PAWS labels are human judgments, and a few are disputable (a pair whose only
  change is a capital letter or a moved space in a name, for example).
- The sentences are shown as PAWS writes them, with spaces around punctuation.
- PAWS is a well-known public dataset, so a model may have seen it while it was
  trained; the labels are not shown, and the bundled pairs are a matched
  subset rather than the split's own order.
- One kind of judgment only: whether reordered words keep their roles.

## Layout

```text
.
├── nanoagent.yaml   # manifest: id, runtime, permissions (no network), tools, the judge-pairs task
├── contract.py      # typed inputs and outputs: BatchInput, PairsOutput, PinOutput, JudgeOutput
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # empty: the plugin needs no environment variables
├── src/plugin/      # the MCP server (__main__.py), the tools (pairs.py), the snapshot and its batches (snapshot.py, snapshot.json.gz, PAWS-LICENSE.txt)
├── tests/           # a made-up snapshot in the bundled file's format; tests marked bundled read the real one
├── ui/              # outputs are drawn from their JSON Schema
├── product/         # the product page: product.yaml and icon.svg
└── verifier/        # deterministic checks: judging.py (the task), pairs.py and pin.py (the tools), rules.py
```

`verifier/rules.py` restates the plugin's fixed rules (the pinned hash, the
number of batches, the batch composition, the sentence checks, `changes` and
`same_words`) because the verifier may not import the plugin's own code;
`tests/test_rules.py` checks that both copies agree.

## Develop and run

```bash
uv sync --extra dev
uv run pytest                              # offline; includes the real bundled snapshot
uv run pytest -m bundled                   # only the tests on the real snapshot
uv run ruff check .
```

Publishing checks every plugin rule: push the plugin to a public GitHub
repository and submit its URL on RUSH Lab's Marketplace.
