"""Prompt builders for the three Self-Refine roles.

Paper mapping (Madaan et al., section 2, Algorithm 1):

- ``build_init_prompt``     -> p_gen:    generate the initial output y0.
- ``build_feedback_prompt`` -> p_fb:     critique y_t with *specific,
  actionable, multi-aspect* feedback (the paper's ablations in section 4 show
  generic feedback like "improve this" barely helps — specificity is the point).
- ``build_refine_prompt``   -> p_refine: rewrite y_t using the feedback.

Every prompt embeds ``<<SELFREFINE:TASK_ID:...>>`` and ``[PHASE: ...]``
markers so :class:`~selfrefine.llm.ScriptedMockLM` can route offline; real
backends ignore them.
"""

from __future__ import annotations

from typing import List

from .feedback import Feedback


def _header(task_id: str, phase: str) -> str:
    return f"<<SELFREFINE:TASK_ID:{task_id}>>\n[PHASE: {phase}]\n"


def build_init_prompt(task_id: str, task_description: str) -> str:
    """Prompt the model to produce its first attempt at the task (p_gen)."""
    return (
        _header(task_id, "INIT")
        + "You are solving the task below. Give your best single attempt.\n\n"
        f"TASK:\n{task_description}\n\n"
        "OUTPUT:\n"
    )


def build_feedback_prompt(
    task_id: str, task_description: str, output: str, iteration: int
) -> str:
    """Prompt the model to critique its own output (p_fb).

    Asks for multi-aspect feedback with a 0-10 score per aspect, mirroring the
    paper's per-aspect scalar scores, and insists on *specific, actionable*
    comments — the paper's key ablation finding.
    """
    return (
        _header(task_id, "FEEDBACK")
        + "You are a strict critic reviewing your own previous output.\n\n"
        f"TASK:\n{task_description}\n\n"
        f"YOUR OUTPUT (iteration {iteration}):\n{output}\n\n"
        "Give feedback as one line per aspect in EXACTLY this format:\n"
        "ASPECT: <name> | SCORE: <0-10>/10 | <comment>\n"
        "Then a final line: VERDICT: NEEDS_REFINEMENT or VERDICT: DONE\n\n"
        "Rules for the comments:\n"
        "- Be SPECIFIC and ACTIONABLE: name the exact flaw and how to fix it\n"
        "  (e.g. 'the loop is O(n^2); use a hash set instead'), never generic\n"
        "  praise or vague advice like 'improve clarity'.\n"
        "- Score honestly: 10 means nothing left to fix on that aspect.\n"
        "- VERDICT: DONE only if every aspect is already at 10/10 quality.\n"
    )


def build_refine_prompt(
    task_id: str,
    task_description: str,
    output: str,
    feedback: Feedback,
    iteration: int,
) -> str:
    """Prompt the model to rewrite its output using the feedback (p_refine)."""
    feedback_lines: List[str] = []
    for aspect in feedback.aspects:
        feedback_lines.append(
            f"ASPECT: {aspect.name} | SCORE: {aspect.score:g}/10 | {aspect.comment}"
        )
    feedback_block = "\n".join(feedback_lines) or "(no structured feedback)"
    return (
        _header(task_id, "REFINE")
        + "You are refining your previous output using the feedback below.\n\n"
        f"TASK:\n{task_description}\n\n"
        f"YOUR PREVIOUS OUTPUT (iteration {iteration}):\n{output}\n\n"
        f"FEEDBACK ON IT:\n{feedback_block}\n\n"
        "Rewrite the output to address EVERY point of feedback. "
        "Keep what was already good. Output ONLY the revised result, no commentary.\n\n"
        "REFINED OUTPUT:\n"
    )


__all__: List[str] = [
    "build_init_prompt",
    "build_feedback_prompt",
    "build_refine_prompt",
]
