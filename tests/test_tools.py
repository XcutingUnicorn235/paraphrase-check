from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import pytest
from conftest import FIXTURE_BATCHES, SAME_COUNTS, fixture_snapshot, gzipped

from contract import BatchInput, Change, Pair, PairsOutput, Slot
from plugin import pin_pairs, read_pairs, snapshot


def test_read_pairs_shows_one_whole_batch_in_order(tmp_path: Path) -> None:
    output = read_pairs(BatchInput(batch=2), tmp_path)
    assert output.batch == 2 and [pair.slot for pair in output.pairs] == list(Slot)
    for pair in output.pairs:
        assert pair.same_words == snapshot.same_words(pair.a, pair.b)
        assert [(change.a, change.b) for change in pair.changes] == snapshot.changes(pair.a, pair.b)
    assert not list(tmp_path.iterdir()), "reading a batch writes nothing"


def test_changes_list_what_moved_and_what_was_swapped() -> None:
    assert snapshot.changes("In 1997 , Anna met Ben .", "Anna met Ben in 1997 .") == [
        ("In 1997 ,", ""),
        ("", "in 1997"),
    ]
    # A swap shows as the longest stretch kept and the rest moved around it.
    assert snapshot.changes("Anna met Ben in Paris .", "Ben met Anna in Paris .") == [("", "Ben met"), ("met Ben", "")]
    assert snapshot.changes("Anna met Ben .", "Anna met Ben .") == []


def test_same_words_counts_repeats_and_ignores_order() -> None:
    assert snapshot.same_words("Anna met Ben .", "Ben met Anna .")
    assert not snapshot.same_words("Anna met Anna .", "Anna met Ben .")
    assert not snapshot.same_words("In 1997 , Anna met Ben .", "Anna met Ben in 1997 .")


def test_read_pairs_never_shows_the_label_or_the_id(tmp_path: Path) -> None:
    assert [field.name for field in fields(PairsOutput)] == ["batch", "pairs"]
    assert [field.name for field in fields(Pair)] == ["slot", "a", "b", "same_words", "changes"]
    text = repr(read_pairs(BatchInput(batch=1), tmp_path))
    assert "same=" not in text and "pair_id" not in text and "PAWS" not in text


def test_read_pairs_holds_no_int_but_the_batch(tmp_path: Path) -> None:
    # With the decide runner every int in a tool's output is a value the output's batch could be copied from.
    shown = read_pairs(BatchInput(batch=3), tmp_path)
    ints = [shown.batch]
    for pair in shown.pairs:
        ints += [v for v in vars(pair).values() if isinstance(v, int) and not isinstance(v, bool)]
    assert ints == [3]
    assert all(isinstance(change, Change) for pair in shown.pairs for change in pair.changes)


def test_pin_pairs_has_the_fixed_composition(tmp_path: Path) -> None:
    for batch in range(FIXTURE_BATCHES):
        keys = pin_pairs(BatchInput(batch=batch), tmp_path).keys
        assert sum(entry.same for entry in keys) == SAME_COUNTS[batch]
        assert [entry.slot for entry in keys] == list(Slot)


def test_batches_are_fixed(tmp_path: Path) -> None:
    def every():
        return [[k.pair_id for k in pin_pairs(BatchInput(batch=b), tmp_path).keys] for b in range(FIXTURE_BATCHES)]

    first = every()
    snapshot.load.cache_clear()
    assert every() == first
    ids = [pair_id for batch in first for pair_id in batch]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("batch", [-1, FIXTURE_BATCHES])
def test_refuses_a_batch_out_of_range(tmp_path: Path, batch: int) -> None:
    with pytest.raises(ValueError, match="batch must be from 0"):
        read_pairs(BatchInput(batch=batch), tmp_path)


def test_refuses_a_changed_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    changed = gzipped(fixture_snapshot(batches=FIXTURE_BATCHES - 1))
    monkeypatch.setattr(snapshot, "read_bytes", lambda: changed)
    with pytest.raises(RuntimeError, match="changed"):
        pin_pairs(BatchInput(batch=0), tmp_path)


def test_refuses_a_snapshot_of_the_wrong_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = gzipped(fixture_snapshot(batches=FIXTURE_BATCHES - 1))
    monkeypatch.setattr(snapshot, "read_bytes", lambda: data)
    monkeypatch.setattr(snapshot, "SNAPSHOT_SHA256", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="batches, not"):
        read_pairs(BatchInput(batch=0), tmp_path)
