"""judge-pairs: the submitted verdicts against PAWS's human labels.

The verdict is valid only when all 4 pairs are judged right. The score is chance-corrected accuracy,
max(0, (accuracy - c) / (1 - c)), where c is the accuracy of the best constant answer for the batch:
the share of its larger kind, 2/4 or 3/4. Answering same everywhere, different everywhere, every
verdict turned around, or at random scores about 0, and only telling the pairs apart earns credit.
Plain accuracy would give a constant answer 1/2 to 3/4.
"""

from contract import Meaning

from . import rules
from .checks import constraint, failed, result
from .pin import key_constraints


def chosen(output):
    return [output.first.p1, output.second.p2, output.third.p3, output.fourth.p4]


def verify_task(arguments, output, reference, workspace):
    # Gates: a broken answer key or an answer to another batch is not compared at all.
    gates = key_constraints(arguments.batch, reference)
    gates.append(
        constraint(
            "output-batch",
            "The output answers the task's batch",
            output.batch == arguments.batch,
            arguments.batch,
            output.batch,
        )
    )
    broken = failed(gates)
    if broken:
        return result(False, broken[0]["constraint"] + ": it does not hold", score=0.0, constraints=gates)

    given = chosen(output)
    labels = [entry.same for entry in reference.keys]
    truth = [Meaning.SAME if same else Meaning.DIFFERENT for same in labels]
    rows = [
        constraint(
            f"{slot.value}-right",
            f"Pair {slot.value} is judged same when its sentences mean the same, and different when not",
            verdict == right,
            right.value,
            verdict.value,
        )
        for slot, verdict, right in zip(rules.SLOTS, given, truth)
    ]
    right = sum(row["ok"] for row in rows)
    accuracy = right / len(rows)
    chance = rules.best_constant_accuracy(labels)
    score = max(0.0, (accuracy - chance) / (1.0 - chance))
    same = [i for i, label in enumerate(labels) if label]
    different = [i for i, label in enumerate(labels) if not label]
    message = (
        f"{right} of 4 pairs right: {sum(rows[i]['ok'] for i in same)} of {len(same)} same-meaning pairs "
        f"and {sum(rows[i]['ok'] for i in different)} of {len(different)} different-meaning pairs"
    )
    return result(
        right == len(rows),
        message,
        score=round(score, 4),
        constraints=gates + rows,
        right=right,
        accuracy=round(accuracy, 4),
        best_constant_accuracy=round(chance, 4),
    )
