"""The verifier restates the plugin's rules; they must agree."""

from __future__ import annotations

import random
from pathlib import Path

import plugin.snapshot as snapshot_module
import verifier.rules as rules_module
from contract import Slot

# The conftest swaps in fixture values; the published ones are compared here, as written in the files.
PUBLISHED = {"SNAPSHOT_SHA256", "BATCHES"}


def published(module, name):
    text = Path(module.__file__).read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith(f"{name} = "))
    return line.split("=", 1)[1].strip().strip('"')


def test_published_constants_agree() -> None:
    for name in PUBLISHED:
        assert published(snapshot_module, name) == published(rules_module, name)
    for name in ("PAIRS_PER_BATCH", "MIN_SAME", "MAX_SAME", "MAX_CHARS"):
        assert getattr(snapshot_module, name) == getattr(rules_module, name), name
    assert rules_module.SLOTS == list(Slot)


def sentences(rng: random.Random, count: int):
    alphabet = ["Anna", "met", "Ben", "in", "Paris", "1997", ",", ".", "the", "of", "é", "Ben", "in"]
    for _ in range(count):
        yield (
            " ".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10))),
            " ".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10))),
        )


def test_changes_and_same_words_agree() -> None:
    for a, b in sentences(random.Random(1), 500):
        assert snapshot_module.changes(a, b) == rules_module.changes(a, b)
        assert snapshot_module.same_words(a, b) == rules_module.same_words(a, b)


def test_sentence_checks_agree() -> None:
    for text in ["", " ", "Anna met Ben .", " Anna", "Anna ", "x" * 300, "x" * 301]:
        assert snapshot_module.sentence_problem(text) == rules_module.sentence_problem(text)


def test_the_composition_keeps_the_best_constant_below_one() -> None:
    rng = random.Random(2)
    for _ in range(300):
        labels = [rng.random() < 0.5 for _ in range(4)]
        if rules_module.composition_holds(labels):
            assert rules_module.best_constant_accuracy(labels) in (0.5, 0.75)
        else:
            assert sum(labels) in (0, 4)
