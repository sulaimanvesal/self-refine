"""LLM backends for Self-Refine.

The paper's core setup (Madaan et al., NeurIPS 2023, section 2): a *single* LLM
plays all three roles — generator, feedback provider, and refiner. So this
module defines one tiny interface, ``LLMBackend.complete(prompt) -> str``, and
two implementations of it:

- :class:`ScriptedMockLM` — deterministic, offline, seeded stand-in used by
  the demo and tests. It only ever sees prompt *text* (exactly like a real
  backend), never the task object.
- :class:`OpenAILM` — optional OpenAI-compatible backend for real runs
  (needs network + an API key).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

# Markers embedded in prompts built by selfrefine.prompts. The mock LM reads
# these to pick its scripted response — a real backend simply ignores them.
TASK_MARKER_RE = re.compile(r"<<SELFREFINE:TASK_ID:([a-z0-9\-]+)>>")
PHASE_MARKER_RE = re.compile(r"\[PHASE:\s*(INIT|FEEDBACK|REFINE)\s*\]")


class LLMBackend(Protocol):
    """The single LLM used as generator, critic, and refiner."""

    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...


@dataclass
class ScriptedMockLM:
    """Deterministic offline stand-in for an LLM.

    ``script`` maps ``task_id`` to a plan::

        {
            "math-01": {
                "init":     "<initial output y0>",
                "feedback": ["<feedback on y0>", "<feedback on y1>", ...],
                "refine":   ["<refined output y1>", "<refined output y2>", ...],
            },
            ...
        }

    Each ``complete()`` call parses the ``<<SELFREFINE:TASK_ID:...>>`` and
    ``[PHASE: ...]`` markers from the prompt text and returns the scripted
    response for that (task, phase, iteration). Iteration counters advance
    independently per phase; if a loop runs longer than the scripted rungs,
    the last rung repeats (the orchestrator's ``max_iters`` still bounds it).
    """

    script: Dict[str, Dict[str, object]]
    _counters: Dict[tuple, int] = field(default_factory=dict, init=False, repr=False)

    def complete(self, prompt: str) -> str:
        task_match = TASK_MARKER_RE.search(prompt)
        phase_match = PHASE_MARKER_RE.search(prompt)
        if not task_match or not phase_match:
            raise ValueError(
                "Prompt is missing the Self-Refine task/phase markers. "
                "Build prompts with selfrefine.prompts."
            )
        task_id = task_match.group(1)
        phase = phase_match.group(1).lower()
        try:
            plan = self.script[task_id]
        except KeyError:
            raise ValueError(f"No scripted plan for task {task_id!r}.")

        if phase == "init":
            response = plan["init"]
        else:
            rungs = plan[phase]
            if not isinstance(rungs, list) or not rungs:
                raise ValueError(f"Script for task {task_id!r} has no {phase!r} rungs.")
            key = (task_id, phase)
            idx = self._counters.get(key, 0)
            response = rungs[min(idx, len(rungs) - 1)]
            self._counters[key] = idx + 1

        if not isinstance(response, str):
            raise TypeError(f"Scripted response for {task_id}/{phase} is not a string.")
        return response

    def reset(self) -> None:
        """Reset per-phase iteration counters (fresh run)."""
        self._counters.clear()


@dataclass
class OpenAILM:
    """Optional OpenAI-compatible backend for real (non-mock) runs.

    Works with OpenAI or any OpenAI-compatible endpoint (set ``base_url``).
    Stdlib only — no extra dependencies.
    """

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    temperature: float = 0.7
    timeout: int = 120

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                "OpenAILM needs an api_key. For offline runs use ScriptedMockLM."
            )
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat-completions response: {data!r}") from exc


__all__: List[str] = [
    "LLMBackend",
    "ScriptedMockLM",
    "OpenAILM",
    "TASK_MARKER_RE",
    "PHASE_MARKER_RE",
]
