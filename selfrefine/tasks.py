"""Demo tasks with scripted LLM behavior for 100% offline runs.

Each task pairs a natural-language description with a scripted "improvement
ladder" for :class:`~selfrefine.llm.ScriptedMockLM`: a flawed first draft,
specific actionable feedback about it, and the refined output the feedback
describes. The ladders mirror the paper's evaluated task families (math
reasoning, code readability, sentiment reversal) and its headline finding:
specific feedback fixes concrete flaws that direct generation misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .llm import ScriptedMockLM


@dataclass
class Task:
    """A task description plus the mock LM's scripted plan for it."""

    id: str
    kind: str
    title: str
    description: str
    script: Dict[str, object] = field(default_factory=dict)


def builtin_tasks() -> List[Task]:
    """Three scripted tasks spanning the paper's task families."""
    return [
        Task(
            id="math-01",
            kind="math reasoning",
            title="Bakery arithmetic",
            description=(
                "A bakery made 240 loaves in the morning and 180 loaves in the "
                "afternoon. By evening they had sold 3/4 of the total. "
                "How many loaves remain? Show your work."
            ),
            script={
                "init": (
                    "Total = 240 + 180 = 420 loaves.\n"
                    "Sold = 3/4 x 420 = 310 loaves.\n"
                    "Remaining = 420 - 310 = 110 loaves."
                ),
                "feedback": [
                    "ASPECT: correctness | SCORE: 4/10 | Arithmetic slip: 3/4 x 420 = 315, "
                    "not 310. Recompute the sold quantity as 420 * 3 / 4 = 315, then "
                    "recompute remaining as 420 - 315.\n"
                    "ASPECT: clarity | SCORE: 8/10 | The step-by-step layout is clear; "
                    "keep the same structure in the revision.\n"
                    "VERDICT: NEEDS_REFINEMENT"
                ],
                "refine": [
                    "Total = 240 + 180 = 420 loaves.\n"
                    "Sold = 3/4 x 420 = 420 * 3 / 4 = 315 loaves.\n"
                    "Remaining = 420 - 315 = 105 loaves."
                ],
            },
        ),
        Task(
            id="code-readability-01",
            kind="code readability",
            title="Readable even-squares",
            description=(
                "Write a Python function that returns the sum of squares of the "
                "even numbers in a list. It must be correct AND easy to read: "
                "clear names, a docstring, and clean formatting."
            ),
            script={
                "init": (
                    "def f(l):\n"
                    "    s=0\n"
                    "    for x in l:\n"
                    "        if x%2==0:s+=x*x\n"
                    "    return s"
                ),
                "feedback": [
                    "ASPECT: readability | SCORE: 3/10 | Single-letter names `f`, `l`, "
                    "`x`, `s` say nothing; rename them to `sum_even_squares`, `numbers`, "
                    "`num`, `total`. Add a docstring describing the argument and return "
                    "value. Put the `if` body on its own indented line instead of "
                    "cramming `if x%2==0:s+=x*x` onto one line.\n"
                    "ASPECT: correctness | SCORE: 9/10 | Logic is right for ints; "
                    "keep the algorithm identical, only change presentation.\n"
                    "VERDICT: NEEDS_REFINEMENT"
                ],
                "refine": [
                    'def sum_even_squares(numbers):\n'
                    '    """Return the sum of squares of the even numbers in `numbers`."""\n'
                    "    total = 0\n"
                    "    for num in numbers:\n"
                    "        if num % 2 == 0:\n"
                    "            total += num * num\n"
                    "    return total"
                ],
            },
        ),
        Task(
            id="sentiment-reversal-01",
            kind="sentiment reversal",
            title="Flip the review",
            description=(
                "Rewrite the following positive restaurant review as a NEGATIVE one, "
                "keeping the same sentence structure:\n"
                "\"The staff were wonderful and the food was absolutely delicious. "
                "I can't wait to come back!\""
            ),
            script={
                "init": (
                    "The staff were terrible and the food was absolutely delicious. "
                    "I can't wait to never come back!"
                ),
                "feedback": [
                    "ASPECT: sentiment-consistency | SCORE: 4/10 | The phrase "
                    "\"absolutely delicious\" still praises the food — replace it with a "
                    "negative descriptor like \"absolutely inedible\". The staff and "
                    "closing-sentence reversals are correct; keep them.\n"
                    "ASPECT: fluency | SCORE: 8/10 | Reads naturally; keep the same "
                    "sentence structure.\n"
                    "VERDICT: NEEDS_REFINEMENT"
                ],
                "refine": [
                    "The staff were terrible and the food was absolutely inedible. "
                    "I can't wait to never come back!"
                ],
            },
        ),
    ]


def feedback_done(task: Task) -> str:
    """The scripted 'all good' feedback the mock returns after the ladder ends."""
    aspects = {
        "math-01": "correctness | SCORE: 10/10 | Arithmetic is now exact: 315 sold, 105 remain.",
        "code-readability-01": "readability | SCORE: 9/10 | Names, docstring, and formatting are all clear.",
        "sentiment-reversal-01": "sentiment-consistency | SCORE: 10/10 | Every clause is now negative.",
    }
    aspect_line = aspects.get(task.id, "quality | SCORE: 10/10 | Nothing left to fix.")
    return f"ASPECT: {aspect_line}\nVERDICT: DONE"


def scripted_mock(tasks: List[Task] | None = None) -> ScriptedMockLM:
    """Build a :class:`ScriptedMockLM` whose ladders end with a DONE verdict.

    Each task's ``feedback`` ladder gets the task-specific DONE feedback
    appended, so the loop converges exactly when the scripted improvements
    run out.
    """
    tasks = tasks if tasks is not None else builtin_tasks()
    script: Dict[str, Dict[str, object]] = {}
    for task in tasks:
        plan = dict(task.script)
        feedback_rungs = list(plan.get("feedback", []))
        feedback_rungs.append(feedback_done(task))
        plan["feedback"] = feedback_rungs
        script[task.id] = plan
    return ScriptedMockLM(script=script)


__all__: List[str] = ["Task", "builtin_tasks", "feedback_done", "scripted_mock"]
