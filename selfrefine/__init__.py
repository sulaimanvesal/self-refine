"""Self-Refine: iterative refinement with self-feedback.

A runnable Python implementation of Madaan et al., "Self-Refine: Iterative
Refinement with Self-Feedback" (NeurIPS 2023, arXiv:2303.17651).

Quick start (offline, no API key)::

    from selfrefine import SelfRefine, builtin_tasks, scripted_mock

    tasks = builtin_tasks()
    task = tasks[0]
    trace = SelfRefine(scripted_mock(tasks)).run(task.id, task.description)
    print(trace.initial_output, "->", trace.final_output)
"""

from .feedback import (
    Aspect,
    Feedback,
    feedback_specificity,
    needs_refinement,
    parse_feedback,
)
from .llm import LLMBackend, OpenAILM, ScriptedMockLM
from .prompts import build_feedback_prompt, build_init_prompt, build_refine_prompt
from .refiner import Iteration, RefinementTrace, SelfRefine
from .tasks import Task, builtin_tasks, scripted_mock

__all__ = [
    "Aspect",
    "Feedback",
    "Iteration",
    "LLMBackend",
    "OpenAILM",
    "RefinementTrace",
    "ScriptedMockLM",
    "SelfRefine",
    "Task",
    "build_feedback_prompt",
    "build_init_prompt",
    "build_refine_prompt",
    "builtin_tasks",
    "feedback_specificity",
    "needs_refinement",
    "parse_feedback",
    "scripted_mock",
]

__version__ = "0.1.0"
