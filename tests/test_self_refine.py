"""Tests for the Self-Refine implementation (offline, no network)."""

import pytest

from selfrefine import (
    SelfRefine,
    builtin_tasks,
    feedback_specificity,
    needs_refinement,
    parse_feedback,
    scripted_mock,
)
from selfrefine.feedback import Aspect, Feedback
from selfrefine.llm import OpenAILM, ScriptedMockLM
from selfrefine.prompts import (
    build_feedback_prompt,
    build_init_prompt,
    build_refine_prompt,
)


# ---------------------------------------------------------------- feedback ---

SPECIFIC = (
    "ASPECT: correctness | SCORE: 4/10 | Arithmetic slip: 3/4 x 420 = 315, "
    "not 310. Recompute 420 * 3 / 4.\n"
    "ASPECT: clarity | SCORE: 8/10 | Keep the step-by-step layout.\n"
    "VERDICT: NEEDS_REFINEMENT"
)

GENERIC = (
    "ASPECT: quality | SCORE: 5/10 | Could be better, try to improve it.\n"
    "VERDICT: NEEDS_REFINEMENT"
)


def test_parse_feedback_extracts_aspects_and_verdict():
    fb = parse_feedback(SPECIFIC)
    assert [a.name for a in fb.aspects] == ["correctness", "clarity"]
    assert fb.aspects[0].score == 4.0
    assert "315" in fb.aspects[0].comment
    assert fb.verdict == "NEEDS_REFINEMENT"


def test_parse_feedback_done_verdict():
    fb = parse_feedback("ASPECT: q | SCORE: 10/10 | perfect.\nVERDICT: DONE")
    assert fb.verdict == "DONE"
    assert not needs_refinement(fb)


def test_parse_feedback_skips_malformed_lines():
    fb = parse_feedback("hello world\nASPECT: a | SCORE: 7/10 | ok\nVERDICT: DONE")
    assert len(fb.aspects) == 1
    assert fb.verdict == "DONE"


def test_parse_feedback_missing_verdict_defaults_to_refine():
    fb = parse_feedback("ASPECT: a | SCORE: 4/10 | needs work")
    assert fb.verdict == "NEEDS_REFINEMENT"
    assert needs_refinement(fb)  # missing verdict + low score -> keep iterating


def test_parse_feedback_empty_text_needs_refinement():
    fb = parse_feedback("")
    assert fb.aspects == []
    assert fb.average_score == 0.0
    assert needs_refinement(fb)


def test_needs_refinement_threshold():
    low = Feedback(aspects=[Aspect("a", 7.9, "x")], verdict="NEEDS_REFINEMENT")
    high = Feedback(aspects=[Aspect("a", 8.0, "x")], verdict="NEEDS_REFINEMENT")
    assert needs_refinement(low, threshold=8.0)
    assert not needs_refinement(high, threshold=8.0)


def test_feedback_specificity_specific_beats_generic():
    specific = parse_feedback(SPECIFIC)
    generic = parse_feedback(GENERIC)
    assert feedback_specificity(specific) > feedback_specificity(generic)
    assert feedback_specificity(generic) == 0.0


def test_feedback_specificity_empty_is_zero():
    assert feedback_specificity(Feedback()) == 0.0


# ------------------------------------------------------------------- mock ---

SCRIPT = {
    "t1": {
        "init": "y0",
        "feedback": ["fb0", "fb1"],
        "refine": ["y1"],
    }
}


def _prompt(task_id="t1", phase="INIT"):
    return f"<<SELFREFINE:TASK_ID:{task_id}>>\n[PHASE: {phase}]\nbody"


def test_mock_routes_init_feedback_refine():
    mock = ScriptedMockLM(script=SCRIPT)
    assert mock.complete(_prompt(phase="INIT")) == "y0"
    assert mock.complete(_prompt(phase="FEEDBACK")) == "fb0"
    assert mock.complete(_prompt(phase="REFINE")) == "y1"


def test_mock_advances_counters_per_phase():
    mock = ScriptedMockLM(script=SCRIPT)
    assert mock.complete(_prompt(phase="FEEDBACK")) == "fb0"
    assert mock.complete(_prompt(phase="FEEDBACK")) == "fb1"
    # rungs exhausted -> last rung repeats (orchestrator max_iters still bounds)
    assert mock.complete(_prompt(phase="FEEDBACK")) == "fb1"


def test_mock_counters_are_independent_per_task_and_phase():
    mock = ScriptedMockLM(script={"a": SCRIPT["t1"], "b": SCRIPT["t1"]})
    mock.complete(_prompt(task_id="a", phase="FEEDBACK"))
    assert mock.complete(_prompt(task_id="b", phase="FEEDBACK")) == "fb0"


def test_mock_reset_restarts_counters():
    mock = ScriptedMockLM(script=SCRIPT)
    mock.complete(_prompt(phase="FEEDBACK"))
    mock.reset()
    assert mock.complete(_prompt(phase="FEEDBACK")) == "fb0"


def test_mock_rejects_prompt_without_markers():
    mock = ScriptedMockLM(script=SCRIPT)
    with pytest.raises(ValueError):
        mock.complete("just some text")


def test_mock_rejects_unknown_task():
    mock = ScriptedMockLM(script=SCRIPT)
    with pytest.raises(ValueError):
        mock.complete(_prompt(task_id="nope", phase="INIT"))


# ----------------------------------------------------------------- prompts ---

def test_prompts_carry_markers_and_content():
    init = build_init_prompt("t1", "do the thing")
    assert "<<SELFREFINE:TASK_ID:t1>>" in init
    assert "[PHASE: INIT]" in init
    assert "do the thing" in init

    fb_prompt = build_feedback_prompt("t1", "do the thing", "y0", 0)
    assert "[PHASE: FEEDBACK]" in fb_prompt
    assert "y0" in fb_prompt
    assert "SPECIFIC" in fb_prompt  # the paper's key instruction

    refine = build_refine_prompt("t1", "do the thing", "y0", parse_feedback(SPECIFIC), 0)
    assert "[PHASE: REFINE]" in refine
    assert "315" in refine  # feedback text is passed through


# ----------------------------------------------------------------- loop -----

def _math_task():
    tasks = builtin_tasks()
    return next(t for t in tasks if t.id == "math-01")


def test_loop_converges_and_refines_math_task():
    tasks = builtin_tasks()
    task = _math_task()
    trace = SelfRefine(scripted_mock(tasks)).run(task.id, task.description)
    assert trace.converged
    assert trace.num_refinements == 1
    assert "110" in trace.initial_output  # the flawed first draft
    assert "105" in trace.final_output  # the corrected answer
    assert trace.final_output != trace.initial_output


def test_loop_stops_at_max_iters_without_converging():
    script = {
        "stubborn": {
            "init": "y0",
            "feedback": ["ASPECT: q | SCORE: 1/10 | bad\nVERDICT: NEEDS_REFINEMENT"],
            "refine": ["y0"],  # never improves
        }
    }
    trace = SelfRefine(ScriptedMockLM(script=script), max_iters=3).run(
        "stubborn", "impossible task"
    )
    assert not trace.converged
    assert trace.num_refinements == 3
    assert len(trace.iterations) == 4


def test_loop_with_zero_max_iters_returns_direct_generation():
    tasks = builtin_tasks()
    task = _math_task()
    trace = SelfRefine(scripted_mock(tasks), max_iters=0).run(task.id, task.description)
    assert trace.num_refinements == 0
    assert trace.final_output == trace.initial_output
    assert not trace.converged


def test_loop_rejects_negative_max_iters():
    with pytest.raises(ValueError):
        SelfRefine(ScriptedMockLM(script={}), max_iters=-1)


def test_score_trajectory_climbs_on_scripted_tasks():
    tasks = builtin_tasks()
    mock = scripted_mock(tasks)
    for task in tasks:
        trace = SelfRefine(mock).run(task.id, task.description)
        scores = [s for s in trace.score_trajectory if s == s]  # drop NaN
        assert scores == sorted(scores), f"{task.id}: scores should not decrease"
        assert scores[-1] > scores[0], f"{task.id}: refinement should improve score"


def test_same_backend_used_for_all_roles():
    # The paper uses ONE llm for generator, critic, and refiner: assert the
    # orchestrator never swaps backends mid-run.
    tasks = builtin_tasks()
    task = _math_task()
    mock = scripted_mock(tasks)
    refiner = SelfRefine(mock)
    refiner.run(task.id, task.description)
    assert refiner.backend is mock


# ------------------------------------------------------------------ tasks ---

def test_builtin_tasks_have_unique_ids_and_scripts():
    tasks = builtin_tasks()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)) >= 3
    for task in tasks:
        assert set(task.script) == {"init", "feedback", "refine"}


def test_scripted_mock_appends_done_verdict():
    tasks = builtin_tasks()
    mock = scripted_mock(tasks)
    for task in tasks:
        rungs = mock.script[task.id]["feedback"]
        assert "VERDICT: DONE" in rungs[-1]


# ------------------------------------------------------------------ openai ---

def test_openai_backend_requires_api_key_before_network():
    backend = OpenAILM(api_key=None)
    with pytest.raises(ValueError, match="api_key"):
        backend.complete("hello")
