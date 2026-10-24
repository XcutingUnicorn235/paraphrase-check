"""The bundled PAWS-Wiki snapshot: read, checked against its pinned hash, and cut into batches.

The snapshot (snapshot.json.gz, next to this file) is a selection of 900 pairs from the test split of
PAWS-Wiki Labeled (Final) (Zhang, Baldridge and He, NAACL 2019; Google LLC; free for any purpose,
notice in PAWS-LICENSE.txt). Every PAWS pair was made by swapping or moving words of a Wikipedia
sentence, or by translating it there and back, and human raters labeled whether the two sentences
still mean the same.

Each batch holds 4 pairs, 1 to 3 of them the same meaning. The pairs were picked so that as many
same-meaning as different-meaning pairs fall in every bin of word overlap, bigram overlap and word
order similarity: those measures do not tell the labels apart. How the selection was made is in
README.md; this module only reads and checks it.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

SNAPSHOT_FILE = Path(__file__).with_name("snapshot.json.gz")
SNAPSHOT_SHA256 = "58ee9520dbaf08a594ecb3f7111dba92d62675d7baab0f05b6f71960ac21aea5"
FORMAT = "paraphrase-check/snapshot/1"
BATCHES = 225
PAIRS_PER_BATCH = 4
MIN_SAME = 1
MAX_SAME = 3
MAX_CHARS = 300


@dataclass(frozen=True)
class Case:
    pair_id: int
    a: str
    b: str
    same: bool


def read_bytes() -> bytes:
    return SNAPSHOT_FILE.read_bytes()


def changes(a: str, b: str) -> list[tuple[str, str]]:
    """Where a and b differ, word by word (words split at spaces), left to right: (a's words, b's words)."""
    x, y = a.split(), b.split()
    matcher = difflib.SequenceMatcher(None, x, y, autojunk=False)
    return [(" ".join(x[i:j]), " ".join(y[k:l])) for tag, i, j, k, l in matcher.get_opcodes() if tag != "equal"]


def same_words(a: str, b: str) -> bool:
    """Whether a and b hold the same words, repeats counted, in any order."""
    return Counter(a.split()) == Counter(b.split())


def sentence_problem(text: str) -> str | None:
    """Why a sentence is unfit to show, or None."""
    if not text.strip():
        return "is empty"
    if text != text.strip():
        return "has leading or trailing spaces"
    if len(text) > MAX_CHARS:
        return f"is longer than {MAX_CHARS} characters"
    return None


def _case(entry: dict) -> Case:
    a, b = entry["a"], entry["b"]
    for text in (a, b):
        problem = sentence_problem(text)
        if problem:
            raise ValueError(f"a sentence of pair {entry['id']} {problem}")
    if a == b:
        raise ValueError(f"pair {entry['id']} shows the same sentence twice")
    if not isinstance(entry["same"], bool):
        raise TypeError(f"pair {entry['id']} has no yes-or-no label")
    return Case(int(entry["id"]), a, b, entry["same"])


def check_batch(number: int, cases: list[Case]) -> None:
    """Raise unless the batch has the fixed shape: 4 distinct pairs, 1 to 3 of them the same meaning."""
    if len(cases) != PAIRS_PER_BATCH or len({case.pair_id for case in cases}) != PAIRS_PER_BATCH:
        raise ValueError(f"batch {number} must hold {PAIRS_PER_BATCH} distinct pairs")
    same = sum(case.same for case in cases)
    if not MIN_SAME <= same <= MAX_SAME:
        raise ValueError(f"batch {number} breaks the fixed composition: {same} of 4 pairs mean the same")


def parse(data: bytes) -> tuple[str, list[list[Case]]]:
    """The snapshot's source note and every batch, each checked for the fixed shape."""
    snapshot = json.loads(gzip.decompress(data).decode("utf-8"))
    if snapshot.get("format") != FORMAT:
        raise ValueError(f"the snapshot's format is {snapshot.get('format')!r}, not {FORMAT!r}")
    batches = [[_case(entry) for entry in batch["pairs"]] for batch in snapshot["batches"]]
    for number, cases in enumerate(batches):
        check_batch(number, cases)
    ids = [case.pair_id for cases in batches for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("a pair appears in more than one batch")
    return snapshot["source"], batches


@cache
def load() -> tuple[str, str, list[list[Case]]]:
    """The checked file's SHA-256, its source note and every batch. Reads once per plugin process."""
    data = read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise RuntimeError(f"the bundled snapshot changed: SHA-256 {digest}, pinned {SNAPSHOT_SHA256}")
    source, batches = parse(data)
    if len(batches) != BATCHES:
        raise ValueError(f"the snapshot holds {len(batches)} batches, not {BATCHES}")
    return digest, source, batches


def batch(number: int) -> tuple[str, str, list[Case]]:
    if not 0 <= number < BATCHES:
        raise ValueError(f"batch must be from 0 to {BATCHES - 1}, not {number}")
    digest, source, batches = load()
    return digest, source, batches[number]
