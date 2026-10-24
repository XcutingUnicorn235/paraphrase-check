"""read_pairs: what the model is shown must be one whole batch, in order, with true hints and no label."""

from . import rules
from .checks import result


def verify_input(value):
    if not rules.in_range(value.batch):
        return result(False, f"batch must be from 0 to {rules.BATCHES - 1}, not {value.batch}")
    return result(True, "the batch exists")


def verify_output(input_value, output_value, workspace):
    if output_value.batch != input_value.batch:
        return result(False, f"read batch {output_value.batch}, not the asked {input_value.batch}")
    if [pair.slot for pair in output_value.pairs] != rules.SLOTS:
        return result(False, "the batch must hold exactly the pairs p1 to p4, in order")
    if len({(pair.a, pair.b) for pair in output_value.pairs}) != rules.PAIRS_PER_BATCH:
        return result(False, "the batch's 4 pairs must be distinct")
    for pair in output_value.pairs:
        name = pair.slot.value
        for side, text in (("A", pair.a), ("B", pair.b)):
            problem = rules.sentence_problem(text)
            if problem:
                return result(False, f"pair {name}'s sentence {side} {problem}")
        if pair.a == pair.b:
            return result(False, f"pair {name} shows the same sentence twice")
        if pair.same_words != rules.same_words(pair.a, pair.b):
            return result(False, f"pair {name}'s same_words does not match its sentences")
        if [(change.a, change.b) for change in pair.changes] != rules.changes(pair.a, pair.b):
            return result(False, f"pair {name}'s changes do not match its sentences")
    return result(True, f"batch {output_value.batch}: 4 pairs, p1 to p4, with their changes")
