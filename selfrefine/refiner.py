"""The Self-Refine orchestration loop (paper Algorithm 1).

Loop::

    y0 = LLM(p_gen)                                  # INIT
    for t in 0..max_iters-1:
        fb_t = LLM(p_fb(task, y_t))                  # FEEDBACK
        if stopping_criterion(fb_t): break           # STOP
        y_{t+1} = LLM(p_refine(task, y_t, fb_t))     # REFINE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .feedback import Feedback, needs_refinement, parse_feedback
from .llm import LLMBackend
from .prompts import (
    build_feedback_prompt,
    build_init_prompt,
    build_refine_prompt,
)


@dataclass
class Iteration:
    """One pass through the loop: an output plus the feedback written about it."""

    index: int  # 0 for the initial output, 1..n for refinements
    output: str
    feedback: Feedback | None = None


@dataclass
class RefinementTrace:
    """Everything the loop produced, for inspection and demoing."""

    task_id: str
    iterations: List[Iteration] = field(default_factory=list)
    converged: bool = False

    @property
    def final_output(self) -> str:
        """The last (best) output."""
        return self.iterations[-1].output

    @property
    def initial_output(self) -> str:
        """The direct-generation baseline (what you'd get without Self-Refine)."""
        return self.iterations[0].output

    @property
    def num_refinements(self) -> int:
        """How many REFINE steps ran (0 means the first draft already passed)."""
        return len(self.iterations) - 1

    @property
    def score_trajectory(self) -> List[float]:
        """Average feedback score per iteration (the paper's quality climb)."""
        return [
            it.feedback.average_score if it.feedback is not None else float("nan")
            for it in self.iterations
        ]


class SelfRefine:
    """Runs the generate -> feedback -> refine loop with one shared LLM.

    Args:
        backend: the single LLM used for all three roles (paper section 2).
        max_iters: max feedback/refine rounds (the paper uses up to 4).
        score_threshold: quality bar per aspect; iteration stops once every
            scored aspect reaches it (paper section 2.3's task-specific
            stopping criterion, made concrete).
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_iters: int = 4,
        score_threshold: float = 8.0,
    ) -> None:
        if max_iters < 0:
            raise ValueError("max_iters must be >= 0")
        self.backend = backend
        self.max_iters = max_iters
        self.score_threshold = score_threshold

    def run(self, task_id: str, task_description: str) -> RefinementTrace:
        """Run Self-Refine on one task; return the full trace."""
        current = self.backend.complete(build_init_prompt(task_id, task_description))
        iterations = [Iteration(index=0, output=current)]
        converged = False

        for t in range(self.max_iters):
            feedback_text = self.backend.complete(
                build_feedback_prompt(task_id, task_description, current, t)
            )
            feedback = parse_feedback(feedback_text)
            iterations[-1].feedback = feedback

            if not needs_refinement(feedback, self.score_threshold):
                converged = True
                break

            current = self.backend.complete(
                build_refine_prompt(task_id, task_description, current, feedback, t)
            )
            iterations.append(Iteration(index=t + 1, output=current))

        return RefinementTrace(
            task_id=task_id, iterations=iterations, converged=converged
        )


__all__: List[str] = ["Iteration", "RefinementTrace", "SelfRefine"]
