"""Offline demo: direct generation vs. Self-Refine on three scripted tasks.

Runs 100% offline with ScriptedMockLM — no API key, no network. Shows the
paper's core result in miniature: the same model, given the chance to critique
and revise its own output, climbs from a flawed first draft to a clean final
answer.
"""

import sys

sys.path.insert(0, ".")

from selfrefine import SelfRefine, builtin_tasks, feedback_specificity, scripted_mock


def main() -> None:
    tasks = builtin_tasks()
    refiner = SelfRefine(scripted_mock(tasks), max_iters=4, score_threshold=8.0)

    print("=" * 72)
    print("SELF-REFINE: iterative refinement with self-feedback")
    print("Madaan et al., NeurIPS 2023 (arXiv:2303.17651) — offline demo")
    print("=" * 72)

    total_gain = 0.0
    for task in tasks:
        trace = refiner.run(task.id, task.description)
        first, last = trace.iterations[0], trace.iterations[-1]
        init_score = first.feedback.average_score if first.feedback else 0.0
        final_score = last.feedback.average_score if last.feedback else 0.0
        total_gain += final_score - init_score

        task_blurb = " ".join(task.description.split())
        print(f"\n### {task.title}  [{task.kind}]")
        print(f"TASK: {task_blurb}")
        print(f"\n  direct generation (y0, score {init_score:.1f}/10):")
        for line in first.output.splitlines():
            print(f"    {line}")
        print(f"\n  self-refined  (y{last.index}, score {final_score:.1f}/10, "
              f"{trace.num_refinements} refinement{'s' if trace.num_refinements != 1 else ''}):")
        for line in last.output.splitlines():
            print(f"    {line}")
        print(f"\n  score trajectory: "
              + " -> ".join(f"{s:.1f}" for s in trace.score_trajectory))
        print(f"  feedback specificity: "
              f"{feedback_specificity(first.feedback):.0%}" if first.feedback else "")
        print(f"  converged: {trace.converged}")

    print("\n" + "=" * 72)
    print(f"average quality gain from self-refinement: +{total_gain / len(tasks):.1f}/10")
    print("the same LLM as generator, critic, and refiner — no training, no reward model")
    print("=" * 72)


if __name__ == "__main__":
    main()
