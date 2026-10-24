"""judge-pairs: the oracle earns full marks, wrong verdicts lose them, and a broken key or batch is refused."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from conftest import FIXTURE_BATCHES, answer, given, key, truth
from slimconfig import build_output, plain_output

import verifier
from contract import BatchInput, PairsOutput, Slot
from plugin import read_pairs

TASK = "judge-pairs"


def verdict(batch, output, reference=None):
    return verifier.verify_task(TASK, given(batch), output, key(batch) if reference is None else reference, "unused")


def failing(result) -> list[str]:
    return [row["id"] for row in result["details"]["constraints"] if not row["ok"]]


def flipped(choice: str) -> str:
    return "different" if choice == "same" else "same"


@pytest.mark.parametrize("batch", range(FIXTURE_BATCHES))
def test_the_key_turned_into_an_answer_scores_full_marks(batch: int) -> None:
    result = verdict(batch, answer(batch, truth(batch)))
    assert result["valid"] is True and result["score"] == 1.0
    assert result["details"]["right"] == 4 and failing(result) == []


def test_one_wrong_pair_in_a_balanced_batch_halves_the_score() -> None:
    choices = truth(0)  # 2 same, 2 different: the best constant is right for 2 of 4
    choices[1] = flipped(choices[1])
    result = verdict(0, answer(0, choices))
    assert result["valid"] is False and result["score"] == 0.5
    assert failing(result) == ["p2-right"]


def test_one_wrong_pair_in_a_lopsided_batch_scores_zero() -> None:
    choices = truth(1)  # 1 same, 3 different: different everywhere is already right for 3 of 4
    choices[0] = flipped(choices[0])
    result = verdict(1, answer(1, choices))
    assert result["valid"] is False and result["score"] == 0.0
    assert result["details"]["best_constant_accuracy"] == 0.75


def test_the_message_counts_both_kinds() -> None:
    result = verdict(2, answer(2, ["same"] * 4))
    assert result["message"] == "3 of 4 pairs right: 3 of 3 same-meaning pairs and 0 of 1 different-meaning pairs"


def test_an_answer_to_another_batch_scores_zero() -> None:
    result = verdict(0, answer(1, truth(0)))
    assert result["valid"] is False and result["score"] == 0.0 and failing(result) == ["output-batch"]


def test_a_missing_key_scores_zero() -> None:
    result = verifier.verify_task(TASK, given(0), answer(0, truth(0)), None, "unused")
    assert result["valid"] is False and result["score"] == 0.0 and failing(result) == ["answer-key"]


def tampered(number: int, **changes):
    return replace(key(number), **changes)


def with_labels(number: int, labels: list[bool]):
    reference = key(number)
    return replace(reference, keys=[replace(entry, same=label) for entry, label in zip(reference.keys, labels)])


@pytest.mark.parametrize(
    ("reference", "gate"),
    [
        (tampered(0, batch=1), "key-batch"),
        (tampered(0, snapshot_sha256="0" * 64), "key-source"),
        (tampered(0, keys=list(reversed(key(0).keys))), "key-pairs"),
        (tampered(0, keys=key(0).keys[:3]), "key-pairs"),
        (tampered(0, keys=[key(0).keys[0]] * 4), "key-pairs"),
        (with_labels(0, [True] * 4), "key-composition"),
        (with_labels(0, [False] * 4), "key-composition"),
    ],
)
def test_a_tampered_key_is_refused(reference, gate) -> None:
    result = verdict(0, answer(0, truth(0)), reference)
    assert result["valid"] is False and result["score"] == 0.0 and gate in failing(result)


def test_the_verdict_is_deterministic_and_small() -> None:
    choices = ["same", "different", "different", "same"]
    first = verdict(3, answer(3, choices))
    second = verdict(3, answer(3, choices))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(json.dumps(first)) < 8000


def test_rows_name_every_pair() -> None:
    result = verdict(0, answer(0, truth(0)))
    rows = [row["id"] for row in result["details"]["constraints"] if row["id"].endswith("-right")]
    assert rows == [f"{slot.value}-right" for slot in Slot]


# The tool checks, on what the sandbox hands them.
def shown(batch: int) -> PairsOutput:
    return build_output(plain_output(read_pairs(given(batch), None)), PairsOutput)


def test_a_real_read_passes_and_out_of_range_batches_are_refused() -> None:
    assert verifier.verify_output("read_pairs", given(2), shown(2), "unused")["valid"] is True
    assert verifier.verify_output("pin_pairs", given(2), key(2), "unused")["valid"] is True
    for tool in ("read_pairs", "pin_pairs"):
        assert verifier.verify_input(tool, given(0))["valid"] is True
        for batch in (-1, FIXTURE_BATCHES):
            assert verifier.verify_input(tool, build_output({"batch": batch}, BatchInput))["valid"] is False


def shown_with(batch: int, change) -> PairsOutput:
    value = plain_output(read_pairs(given(batch), None))
    change(value)
    return build_output(value, PairsOutput)


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (lambda v: v.update(batch=1), "not the asked"),
        (lambda v: v["pairs"].reverse(), "p1 to p4, in order"),
        (lambda v: v["pairs"].pop(), "p1 to p4, in order"),
        (lambda v: v["pairs"][0].update(a="x" * 301), "longer than 300"),
        (lambda v: v["pairs"][0].update(b=" padded"), "leading or trailing"),
        (lambda v: v["pairs"][0].update(b=v["pairs"][0]["a"]), "same sentence twice"),
        (lambda v: v["pairs"][0].update(same_words=not v["pairs"][0]["same_words"]), "same_words"),
        (lambda v: v["pairs"][0]["changes"].pop(), "changes do not match"),
        (lambda v: v["pairs"][1].update(a=v["pairs"][0]["a"], b=v["pairs"][0]["b"]), "distinct"),
    ],
)
def test_a_wrong_read_is_refused(change, words: str) -> None:
    result = verifier.verify_output("read_pairs", given(0), shown_with(0, change), "unused")
    assert result["valid"] is False and words in result["message"]


def test_a_broken_key_is_refused_by_the_pin_check() -> None:
    result = verifier.verify_output("pin_pairs", given(0), with_labels(0, [False] * 4), "unused")
    assert result["valid"] is False and "1 to 3 pairs" in result["message"]
