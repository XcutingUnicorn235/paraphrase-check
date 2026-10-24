"""pin_pairs: the answer key must come from the pinned snapshot and have the batch's fixed shape."""

from . import rules
from .checks import constraint, failed, result


def key_constraints(batch, key):
    """The answer key's own checks: right batch, pinned snapshot, p1 to p4 in order with distinct pair ids,
    and the fixed composition."""
    if key is None:
        return [constraint("answer-key", "The task has an answer key", False, "pinned", "missing")]
    slots = [entry.slot for entry in key.keys]
    ids = [entry.pair_id for entry in key.keys]
    labels = [entry.same for entry in key.keys]
    return [
        constraint("key-batch", "The answer key is for the task's batch", key.batch == batch, batch, key.batch),
        constraint(
            "key-source",
            "The answer key was read from the pinned snapshot",
            key.snapshot_sha256 == rules.SNAPSHOT_SHA256,
            rules.SNAPSHOT_SHA256,
            key.snapshot_sha256,
        ),
        constraint(
            "key-pairs",
            "The answer key has 4 distinct pairs, p1 to p4, in order",
            slots == rules.SLOTS and len(set(ids)) == len(ids),
            [slot.value for slot in rules.SLOTS],
            [slot.value for slot in slots],
        ),
        constraint(
            "key-composition",
            "The batch has 1 to 3 pairs that mean the same, so both kinds are in it",
            rules.composition_holds(labels),
            "1 to 3",
            sum(labels),
        ),
    ]


def verify_input(value):
    if not rules.in_range(value.batch):
        return result(False, f"batch must be from 0 to {rules.BATCHES - 1}, not {value.batch}")
    return result(True, "the batch exists")


def verify_output(input_value, output_value, workspace):
    problems = failed(key_constraints(input_value.batch, output_value))
    if problems:
        return result(False, problems[0]["constraint"] + ": it does not hold", constraints=problems)
    return result(True, f"answer key for batch {input_value.batch} is pinned")
