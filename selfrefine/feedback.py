"""Structured self-feedback: multi-aspect scores and the stopping criterion.

The paper (sections 2.2-2.3) has the model emit feedback with scalar quality
scores per aspect, and stops iterating when a task-specific criterion is met
(e.g. no aspect still needs improvement). This module parses that structured
feedback and implements the stopping check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Lines like:  ASPECT: readability | SCORE: 3/10 | rename the variables ...
_ASPECT_RE = re.compile(
    r"ASPECT:\s*(?P<name>[^|]+?)\s*\|\s*SCORE:\s*(?P<score>\d+(?:\.\d+)?)\s*/\s*10\s*\|\s*(?P<comment>.*)"
)
_VERDICT_RE = re.compile(r"VERDICT:\s*(?P<verdict>[A-Z_]+)")

# Heuristics for the paper's "specific, actionable feedback" ablation (section 4):
# a comment counts as concrete if it names numbers, quoted spans, or code, or
# uses an imperative fix verb.
_CONCRETE_RE = re.compile(r"(\d+|`[^`]+`|\"[^\"]+\"|'[^']+')")
_IMPERATIVE_RE = re.compile(
    r"\b(replace|add|remove|fix|change|use|rename|extract|avoid|include|rewrite|"
    r"correct|recompute|recheck|split|merge|delete|insert)\b",
    re.IGNORECASE,
)


@dataclass
class Aspect:
    """One scored dimension of the feedback, e.g. correctness or readability."""

    name: str
    score: float  # 0..10
    comment: str = ""


@dataclass
class Feedback:
    """Parsed self-feedback for one iteration's output."""

    aspects: List[Aspect] = field(default_factory=list)
    verdict: str = "NEEDS_REFINEMENT"
    raw: str = ""

    @property
    def average_score(self) -> float:
        """Mean aspect score; 0.0 when there are no aspects."""
        if not self.aspects:
            return 0.0
        return sum(a.score for a in self.aspects) / len(self.aspects)

    def low_aspects(self, threshold: float) -> List[Aspect]:
        """Aspects still below the quality bar."""
        return [a for a in self.aspects if a.score < threshold]


def parse_feedback(text: str) -> Feedback:
    """Parse the model's structured feedback text into a :class:`Feedback`.

    Malformed lines are skipped; a missing VERDICT line defaults to
    ``NEEDS_REFINEMENT`` (fail safe: keep iterating rather than stop early).
    """
    aspects: List[Aspect] = []
    verdict = "NEEDS_REFINEMENT"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        aspect_match = _ASPECT_RE.match(line)
        if aspect_match:
            aspects.append(
                Aspect(
                    name=aspect_match.group("name").strip(),
                    score=float(aspect_match.group("score")),
                    comment=aspect_match.group("comment").strip(),
                )
            )
            continue
        verdict_match = _VERDICT_RE.search(line)
        if verdict_match:
            verdict = verdict_match.group("verdict").strip().upper()
    return Feedback(aspects=aspects, verdict=verdict, raw=text)


def needs_refinement(feedback: Feedback, threshold: float = 8.0) -> bool:
    """Stopping criterion (paper section 2.3, made concrete).

    Stop when the model itself declares DONE, or when every scored aspect is
    at/above the quality threshold. With no parseable aspects we keep going —
    an unreadable critique must not silently end the loop.
    """
    if feedback.verdict == "DONE":
        return False
    if not feedback.aspects:
        return True
    return any(aspect.score < threshold for aspect in feedback.aspects)


def feedback_specificity(feedback: Feedback) -> float:
    """Fraction of aspect comments that are concrete/actionable (0..1).

    Quantifies the paper's central ablation: specific feedback ("recompute
    3/4 x 420 = 315") drives refinement; generic feedback ("improve it")
    does not.
    """
    comments = [a.comment for a in feedback.aspects if a.comment.strip()]
    if not comments:
        return 0.0
    concrete = sum(
        1
        for comment in comments
        if _CONCRETE_RE.search(comment) or _IMPERATIVE_RE.search(comment)
    )
    return concrete / len(comments)


__all__: List[str] = [
    "Aspect",
    "Feedback",
    "parse_feedback",
    "needs_refinement",
    "feedback_specificity",
]
