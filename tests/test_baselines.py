"""Trivial agents and a naive verifier, scored on every batch: the fixture's and the real bundled snapshot's.

An agent that cannot tell the pairs apart must score about 0 here. The naive verifier is what a quick
check would use, plain accuracy against the key; the numbers show what it would reward instead.
"""

from __future__ import annotations

import itertools
from statistics import mean

import pytest
from conftest import FIXTURE_BATCHES, answer, given, key, truth

import verifier
from plugin import snapshot

VERDICTS = ["same", "different"]


def scores(agent, batches: int) -> list[float]:
    out = []
    for batch in range(batches):
        result = verifier.verify_task("judge-pairs", given(batch), answer(batch, agent(batch)), key(batch), "unused")
        out.append(result["score"])
    return out


def plain_accuracy(batch: int, verdicts: list[str]) -> float:
    return mean(verdict == right for verdict, right in zip(verdicts, truth(batch)))


def inverted(batch: int) -> list[str]:
    return ["different" if right == "same" else "same" for right in truth(batch)]


def check_trivial_agents(batches: int) -> None:
    for verdict in VERDICTS:
        assert scores(lambda batch, v=verdict: [v] * 4, batches) == [0.0] * batches, verdict
    assert scores(inverted, batches) == [0.0] * batches
    assert scores(truth, batches) == [1.0] * batches
    # A fair coin, scored exactly: every one of the 16 possible answers, equally likely. It is right on
    # all 4 pairs once in 16, and on 3 of a 2-2 batch a quarter of the time, so it scores 0.1875 on a
    # 2-2 batch and 0.0625 on a 1-3 or 3-1 one; with a third of each, about 0.10.
    coin = [mean(scores(lambda batch, v=list(v): v, batches)) for v in itertools.product(VERDICTS, repeat=4)]
    assert mean(coin) == pytest.approx((0.1875 + 2 * 0.0625) / 3)


def test_trivial_agents_score_zero_and_the_oracle_one_on_the_fixture() -> None:
    check_trivial_agents(FIXTURE_BATCHES)


@pytest.mark.bundled
def test_trivial_agents_score_zero_and_the_oracle_one_on_the_real_snapshot() -> None:
    check_trivial_agents(snapshot.BATCHES)


@pytest.mark.bundled
def test_plain_accuracy_would_reward_a_constant_answer() -> None:
    batches = range(snapshot.BATCHES)
    for verdict in VERDICTS:
        accuracies = [plain_accuracy(b, [verdict] * 4) for b in batches]
        assert set(accuracies) == {0.25, 0.5, 0.75} and mean(accuracies) == 0.5, verdict
    best = [max(plain_accuracy(b, [v] * 4) for v in VERDICTS) for b in batches]
    assert mean(best) == pytest.approx(2 / 3)  # a third of the batches each 2-2, 1-3 and 3-1
