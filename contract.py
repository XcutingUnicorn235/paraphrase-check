"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    PIN_PAIRS = "pin_pairs"
    READ_PAIRS = "read_pairs"


class Task(Enum):
    JUDGE_PAIRS = "judge-pairs"


class Slot(Enum):
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"


class Meaning(Enum):
    SAME = "same"
    DIFFERENT = "different"


BATCH = (
    "Which batch of 4 sentence pairs to judge, from 0 to 224; every batch is fixed and mixes pairs that mean "
    "the same and pairs that do not"
)
SLOT = "The pair's place in its batch, p1 to p4"
MEANING = "same when sentences A and B mean the same thing, different when they do not"


@dataclass
class BatchInput(Output):
    batch: Annotated[int, BATCH]


@dataclass
class Change(Output):
    a: Annotated[str, "The words of sentence A at this place, or an empty string when B adds words here"]
    b: Annotated[str, "The words of sentence B at this place, or an empty string when A has words B drops"]


@dataclass
class Pair(Output):
    slot: Annotated[Slot, SLOT]
    a: Annotated[str, "Sentence A, as written in the source (words and punctuation separated by spaces)"]
    b: Annotated[str, "Sentence B, as written in the source"]
    same_words: Annotated[
        bool,
        "Whether A and B use exactly the same words, repeats counted, only in another order: a hint, not a verdict",
    ]
    changes: Annotated[list[Change], "Where A and B differ, word by word from left to right; equal stretches left out"]


@dataclass
class PairsOutput(Output):
    batch: Annotated[int, "The batch number, as asked"]
    pairs: Annotated[list[Pair], "The batch's 4 pairs, p1 to p4, in order"]


@dataclass
class Key(Output):
    slot: Annotated[Slot, SLOT]
    pair_id: Annotated[int, "The pair's id in PAWS-Wiki's test split"]
    same: Annotated[bool, "Whether PAWS's human raters say the two sentences mean the same"]


@dataclass
class PinOutput(Output):
    batch: Annotated[int, "The batch number, as asked"]
    snapshot_sha256: Annotated[str, "SHA-256 of the bundled snapshot file the key was read from"]
    source: Annotated[str, "Where the pairs and their labels come from"]
    keys: Annotated[list[Key], "The batch's answer key, one entry per pair, p1 to p4"]


# Each verdict sits in a level of its own: the decide runner asks each nested level in a request of
# its own, so each request weighs one pair. Asked two at a time, the model often spent its whole
# 2,048-token reply going back and forth on one hard pair before it answered either.
@dataclass
class FirstPair(Output):
    p1: Annotated[Meaning, MEANING]


@dataclass
class SecondPair(Output):
    p2: Annotated[Meaning, MEANING]


@dataclass
class ThirdPair(Output):
    p3: Annotated[Meaning, MEANING]


@dataclass
class FourthPair(Output):
    p4: Annotated[Meaning, MEANING]


@dataclass
class JudgeOutput(Output):
    batch: Annotated[int, "The batch number, as given"]
    first: Annotated[FirstPair, "The verdict for pair p1"]
    second: Annotated[SecondPair, "The verdict for pair p2"]
    third: Annotated[ThirdPair, "The verdict for pair p3"]
    fourth: Annotated[FourthPair, "The verdict for pair p4"]


TOOLS = {
    Tool.PIN_PAIRS: (BatchInput, PinOutput),
    Tool.READ_PAIRS: (BatchInput, PairsOutput),
}
TASKS = {Task.JUDGE_PAIRS: (BatchInput, JudgeOutput)}
