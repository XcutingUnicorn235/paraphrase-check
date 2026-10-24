"""The plugin's fixed rules, restated here because the verifier may not import the plugin's own code.

They must agree with src/plugin/snapshot.py; tests/test_rules.py checks that they do.
"""

import difflib
from collections import Counter

from contract import Slot

SNAPSHOT_SHA256 = "58ee9520dbaf08a594ecb3f7111dba92d62675d7baab0f05b6f71960ac21aea5"
BATCHES = 225
PAIRS_PER_BATCH = 4
MIN_SAME = 1
MAX_SAME = 3
MAX_CHARS = 300
SLOTS = list(Slot)


def in_range(batch):
    return 0 <= batch < BATCHES


def changes(a, b):
    """Where a and b differ, word by word (words split at spaces), left to right: (a's words, b's words)."""
    x, y = a.split(), b.split()
    matcher = difflib.SequenceMatcher(None, x, y, autojunk=False)
    return [(" ".join(x[i:j]), " ".join(y[k:l])) for tag, i, j, k, l in matcher.get_opcodes() if tag != "equal"]


def same_words(a, b):
    """Whether a and b hold the same words, repeats counted, in any order."""
    return Counter(a.split()) == Counter(b.split())


def sentence_problem(text):
    """Why a sentence is unfit to show, or None."""
    if not text.strip():
        return "is empty"
    if text != text.strip():
        return "has leading or trailing spaces"
    if len(text) > MAX_CHARS:
        return f"is longer than {MAX_CHARS} characters"
    return None


def composition_holds(labels):
    """4 labels, MIN_SAME to MAX_SAME of them the same meaning: both kinds in every batch."""
    return len(labels) == PAIRS_PER_BATCH and MIN_SAME <= sum(labels) <= MAX_SAME


def best_constant_accuracy(labels):
    """The accuracy of the best answer that is the same for every pair: the larger kind's share."""
    same = sum(labels)
    return max(same, len(labels) - same) / len(labels)
