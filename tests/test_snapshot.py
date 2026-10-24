"""The snapshot: the fixture parses, a broken snapshot is refused, and the bundled file is the pinned one
with every batch in the fixed shape and the surface measures unable to tell the labels apart."""

from __future__ import annotations

import copy
import difflib
import gzip
import hashlib
import itertools
import json
import re

import pytest
from conftest import fixture_snapshot, gzipped
from slimconfig import build_output, plain_output

import verifier
from contract import BatchInput, PairsOutput
from plugin import read_pairs, snapshot


def test_the_fixture_parses() -> None:
    source, batches = snapshot.parse(gzipped(fixture_snapshot()))
    assert source == "made-up fixture" and len(batches) == 6


def broken(change) -> bytes:
    value = copy.deepcopy(fixture_snapshot())
    change(value["batches"])
    return gzipped(value)


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (lambda bs: bs[0]["pairs"][0].update(b=bs[0]["pairs"][0]["a"]), "same sentence twice"),
        (lambda bs: bs[0]["pairs"][0].update(a="x" * 301), "longer than 300"),
        (lambda bs: bs[0]["pairs"][0].update(a=""), "is empty"),
        (lambda bs: bs[0]["pairs"][0].update(a="Anna met Ben . "), "leading or trailing"),
        (lambda bs: bs[0]["pairs"].pop(), "4 distinct pairs"),
        (lambda bs: bs[0]["pairs"][1].update(id=bs[0]["pairs"][0]["id"]), "4 distinct pairs"),
        (lambda bs: [pair.update(same=True) for pair in bs[0]["pairs"]], "fixed composition"),
        (lambda bs: [pair.update(same=False) for pair in bs[0]["pairs"]], "fixed composition"),
        (lambda bs: bs[1]["pairs"][0].update(id=bs[0]["pairs"][0]["id"]), "more than one batch"),
    ],
)
def test_a_broken_snapshot_is_refused(change, words: str) -> None:
    with pytest.raises(ValueError, match=words):
        snapshot.parse(broken(change))


def test_a_wrong_format_is_refused() -> None:
    value = fixture_snapshot()
    value["format"] = "something/else"
    with pytest.raises(ValueError, match="format"):
        snapshot.parse(gzipped(value))


@pytest.mark.bundled
def test_the_bundled_snapshot_is_the_pinned_one() -> None:
    data = snapshot.read_bytes()
    assert hashlib.sha256(data).hexdigest() == snapshot.SNAPSHOT_SHA256
    _, source, batches = snapshot.load()
    assert len(batches) == snapshot.BATCHES == 225
    assert "PAWS-Wiki" in source and "Google LLC" in source
    cases = [case for batch in batches for case in batch]
    assert len(cases) == 900 and sum(case.same for case in cases) == 450
    sentences = [text for case in cases for text in (case.a, case.b)]
    assert len(set(sentences)) == len(sentences), "every sentence appears once"
    assert json.loads(gzip.decompress(data))["format"] == snapshot.FORMAT


@pytest.mark.bundled
def test_every_bundled_batch_passes_the_read_verifier() -> None:
    for number in range(snapshot.BATCHES):
        given = build_output({"batch": number}, BatchInput)
        shown = build_output(plain_output(read_pairs(given, None)), PairsOutput)
        assert verifier.verify_output("read_pairs", given, shown, "unused")["valid"] is True, number


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def best_balanced_accuracy(values: list[float], labels: list[bool]) -> float:
    """The best threshold rule on one measure, either way round: mean of its two per-label accuracies."""
    same = sum(labels)
    different = len(labels) - same
    best = 0.5
    for t in sorted(set(values)):
        hit = sum(1 for v, y in zip(values, labels) if y and v >= t) / same
        reject = sum(1 for v, y in zip(values, labels) if not y and v < t) / different
        best = max(best, (hit + reject) / 2, 1 - (hit + reject) / 2)
    return best


@pytest.mark.bundled
def test_surface_measures_do_not_tell_the_labels_apart() -> None:
    _, _, batches = snapshot.load()
    cases = [case for batch in batches for case in batch]
    labels = [case.same for case in cases]
    measures = {
        "word overlap": lambda x, y: jaccard(x, y),
        "bigram overlap": lambda x, y: jaccard(list(itertools.pairwise(x)), list(itertools.pairwise(y))),
        "word order": lambda x, y: difflib.SequenceMatcher(None, x, y, autojunk=False).ratio(),
        "length gap": lambda x, y: -abs(len(x) - len(y)),
        "same words": lambda x, y: float(sorted(x) == sorted(y)),
        "changes": lambda x, y: -float(len(snapshot.changes(" ".join(x), " ".join(y)))),
    }
    for name, measure in measures.items():
        values = [round(measure(words(case.a), words(case.b)), 3) for case in cases]
        assert best_balanced_accuracy(values, labels) < 0.56, name


@pytest.mark.bundled
def test_the_license_notice_ships_with_the_data() -> None:
    notice = snapshot.SNAPSHOT_FILE.with_name("PAWS-LICENSE.txt").read_text(encoding="utf-8")
    assert "freely used for any purpose" in notice and "Google LLC" in notice


def test_a_label_that_is_not_yes_or_no_is_refused() -> None:
    with pytest.raises(TypeError, match="yes-or-no label"):
        snapshot.parse(broken(lambda bs: bs[0]["pairs"][0].update(same=1)))
