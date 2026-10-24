"""The two tools: read a batch's pairs (what the model sees), and pin its answer key (what it never sees)."""

from __future__ import annotations

from pathlib import Path

from contract import BatchInput, Change, Key, Pair, PairsOutput, PinOutput, Slot
from plugin import snapshot

SLOTS = list(Slot)


def read_pairs(given: BatchInput, workspace: Path) -> PairsOutput:
    """The 4 pairs with what differs between their sentences: no id, no label."""
    _, _, cases = snapshot.batch(given.batch)
    pairs = [
        Pair(
            slot=slot,
            a=case.a,
            b=case.b,
            same_words=snapshot.same_words(case.a, case.b),
            changes=[Change(a=a, b=b) for a, b in snapshot.changes(case.a, case.b)],
        )
        for slot, case in zip(SLOTS, cases, strict=True)
    ]
    return PairsOutput(batch=given.batch, pairs=pairs)


def pin_pairs(given: BatchInput, workspace: Path) -> PinOutput:
    digest, source, cases = snapshot.batch(given.batch)
    keys = [Key(slot=slot, pair_id=case.pair_id, same=case.same) for slot, case in zip(SLOTS, cases, strict=True)]
    return PinOutput(batch=given.batch, snapshot_sha256=digest, source=source, keys=keys)
