"""Offline fixtures: a small snapshot in the bundled file's own format, made up here, so most tests do not
depend on the real data. Tests marked ``bundled`` read the real bundled snapshot instead."""

from __future__ import annotations

import gzip
import hashlib
import json

import pytest
from slimconfig import build_output, plain_output

from contract import BatchInput, JudgeOutput, PinOutput
from plugin import pin_pairs, snapshot
from verifier import rules

FIXTURE_BATCHES = 6
# How many of each batch's 4 pairs mean the same, batch by batch; the rest swap who did what.
SAME_COUNTS = [2, 1, 3, 2, 3, 1]


def fixture_snapshot(batches: int = FIXTURE_BATCHES) -> dict:
    """Made-up pairs: a phrase moved (same meaning) or two names swapped (different meaning)."""
    out = []
    for b in range(batches):
        same = SAME_COUNTS[b % len(SAME_COUNTS)]
        pairs = []
        for p in range(4):
            year = 1900 + 10 * b + p
            a = f"In {year} , Anna met Ben in Paris ."
            if p < same:
                pairs.append({"id": 1000 * b + p, "a": a, "b": f"Anna met Ben in Paris in {year} .", "same": True})
            else:
                pairs.append({"id": 1000 * b + p, "a": a, "b": f"In {year} , Ben met Anna in Paris .", "same": False})
        # Same-meaning pairs not always first.
        pairs = pairs[b % 4 :] + pairs[: b % 4]
        out.append({"pairs": pairs})
    return {"format": snapshot.FORMAT, "source": "made-up fixture", "batches": out}


def gzipped(value: dict) -> bytes:
    return gzip.compress(json.dumps(value, sort_keys=True).encode("utf-8"), mtime=0)


FIXTURE = gzipped(fixture_snapshot())
FIXTURE_SHA256 = hashlib.sha256(FIXTURE).hexdigest()


def given(batch: int) -> BatchInput:
    return build_output({"batch": batch}, BatchInput)


def key(batch: int) -> PinOutput:
    """The answer key as NanoAgent's sandbox hands it to the verifier: plain JSON built into its class."""
    return build_output(plain_output(pin_pairs(given(batch), None)), PinOutput)


def answer(batch: int, verdicts: list[str]) -> JudgeOutput:
    p1, p2, p3, p4 = verdicts
    return build_output(
        {"batch": batch, "first": {"p1": p1}, "second": {"p2": p2}, "third": {"p3": p3}, "fourth": {"p4": p4}},
        JudgeOutput,
    )


def truth(batch: int) -> list[str]:
    return ["same" if entry.same else "different" for entry in key(batch).keys]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Every test reads the fixture, unless it is marked bundled."""
    snapshot.load.cache_clear()
    if request.node.get_closest_marker("bundled"):
        yield
        snapshot.load.cache_clear()
        return
    monkeypatch.setattr(snapshot, "read_bytes", lambda: FIXTURE)
    monkeypatch.setattr(snapshot, "SNAPSHOT_SHA256", FIXTURE_SHA256)
    monkeypatch.setattr(snapshot, "BATCHES", FIXTURE_BATCHES)
    monkeypatch.setattr(rules, "SNAPSHOT_SHA256", FIXTURE_SHA256)
    monkeypatch.setattr(rules, "BATCHES", FIXTURE_BATCHES)
    yield
    snapshot.load.cache_clear()
