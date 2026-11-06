import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents.evaluator import InvalidEvaluationJSONError
from app.agents.generator import InvalidTopicError
from app.agents.improver import MissingFeedbackError
from app.evaluation.schemas import CRITERIA, EvaluationResult, LessonContent
from app.llm import LLMError
from app.workflow import graph as graph_module
from app.workflow import nodes
from app.workflow.graph import (
    route_after_evaluate,
    route_after_generate,
    route_after_improve,
    run_workflow,
)
from app.workflow.state import MAX_IMPROVEMENT_ATTEMPTS, initial_state
from langgraph.graph import END


def make_lesson(title: str = "v1") -> LessonContent:
    section = {"heading": "H", "explanation": "E", "example": "X"}
    return LessonContent.model_validate(
        {
            "title": title,
            "topic": "RAG",
            "learner_level": "Beginner",
            "learning_objectives": ["a", "b"],
            "sections": [section, section, section],
            "key_takeaways": ["a", "b", "c"],
            "summary": "s",
        }
    )


def make_eval(failing: tuple[int, ...] = ()) -> EvaluationResult:
    checks = [
        {
            "criterion": name,
            "passed": i not in failing,
            "reason": "Because.",
            "improvement_suggestion": "Fix it." if i in failing else "",
            "evidence_quote": "" if i in failing else "ok",
            "issues": (
                [
                    {
                        "problem": "A problem.",
                        "quote": "Intro",
                        "term": "RAG",
                        "correction": "The correct statement.",
                    }
                ]
                if i in failing
                else []
            ),
        }
        for i, name in enumerate(CRITERIA)
    ]
    return EvaluationResult.model_validate(
        {
            "overall_pass": not failing,
            "checks": checks,
            "failure_reasons": [f"{CRITERIA[i]} failed" for i in failing],
            "improvement_feedback": "Do better." if failing else "",
        }
    )


@pytest.fixture
def agents(monkeypatch):
    gen = AsyncMock(return_value=make_lesson("v1"))
    ev = AsyncMock(return_value=make_eval())
    imp = AsyncMock(
        side_effect=lambda lesson, evaluation, **kwargs: make_lesson("improved")
    )
    monkeypatch.setattr(nodes, "generate_lesson", gen)
    monkeypatch.setattr(nodes, "evaluate_lesson", ev)
    monkeypatch.setattr(nodes, "improve_lesson", imp)
    return gen, ev, imp


def run(max_attempts: int = 2) -> dict:
    return asyncio.run(run_workflow("RAG", maximum_attempts=max_attempts))


def test_initial_lesson_passes_immediately(agents):
    gen, _, imp = agents
    state = run()
    assert state["status"] == "passed"
    assert state["improvement_attempts"] == 0
    assert state["evaluation_count"] == 1
    assert state["rejection_log"] == []
    assert state["lesson"]["title"] == "v1"
    imp.assert_not_awaited()
    gen.assert_awaited_once_with("RAG")


def test_first_fails_then_improvement_passes(agents):
    _, ev, _ = agents
    ev.side_effect = [make_eval((1,)), make_eval()]
    state = run()
    assert state["status"] == "passed"
    assert state["improvement_attempts"] == 1
    assert state["evaluation_count"] == 2
    assert state["lesson"]["title"] == "improved"
    assert state["evaluation"]["overall_pass"] is True
    assert len(state["rejection_log"]) == 1


def test_second_improvement_passes(agents):
    _, ev, _ = agents
    ev.side_effect = [make_eval((1,)), make_eval((2,)), make_eval()]
    state = run()
    assert state["status"] == "passed"
    assert state["improvement_attempts"] == 2
    assert state["evaluation_count"] == 3
    assert len(state["rejection_log"]) == 2


def test_all_attempts_fail_terminates_rejected(agents):
    _, ev, imp = agents
    ev.return_value = make_eval((0,))
    state = run()
    assert state["status"] == "rejected"
    assert state["improvement_attempts"] == 2
    assert state["evaluation_count"] == 3
    assert imp.await_count == 2
    assert state["evaluation"]["overall_pass"] is False
    assert state["error"] is None


def test_attempts_never_exceed_limit_even_if_config_is_larger(agents):
    _, ev, imp = agents
    ev.return_value = make_eval((0,))
    state = run(max_attempts=99)
    assert state["maximum_attempts"] == MAX_IMPROVEMENT_ATTEMPTS == 2
    assert state["improvement_attempts"] == 2
    assert imp.await_count == 2


def test_zero_attempts_rejects_without_improving(agents):
    _, ev, imp = agents
    ev.return_value = make_eval((0,))
    state = run(max_attempts=0)
    assert state["status"] == "rejected"
    assert state["improvement_attempts"] == 0
    imp.assert_not_awaited()
    assert state["rejection_log"][0]["another_attempt_made"] is False


def test_routing_functions():
    base = initial_state("RAG", 2)
    assert route_after_generate(base) == "evaluate"
    assert route_after_generate({**base, "status": "error"}) == END
    assert route_after_evaluate({**base, "status": "in_progress"}) == "improve"
    assert route_after_evaluate({**base, "status": "passed"}) == END
    assert route_after_evaluate({**base, "status": "rejected"}) == END
    assert route_after_evaluate({**base, "status": "error"}) == END
    assert route_after_improve(base) == "evaluate"
    assert route_after_improve({**base, "status": "error"}) == END


def test_rejection_log_accuracy(agents):
    _, ev, _ = agents
    ev.side_effect = [make_eval((1, 3)), make_eval((4,)), make_eval((4,))]
    state = run()
    first, second, third = (
        state["rejection_log"][0],
        state["rejection_log"][1],
        None,
    )
    assert len(state["rejection_log"]) == 3
    third = state["rejection_log"][2]

    assert first["attempt_number"] == 1
    assert first["failed_criteria"] == [CRITERIA[1], CRITERIA[3]]
    assert first["failure_reasons"] == [
        f"{CRITERIA[1]} failed",
        f"{CRITERIA[3]} failed",
    ]
    assert first["improvement_feedback"] == "Do better."
    assert first["another_attempt_made"] is True
    assert "title changed" in first["changes"]

    assert second["attempt_number"] == 2
    assert second["failed_criteria"] == [CRITERIA[4]]
    assert second["another_attempt_made"] is True

    assert third["attempt_number"] == 3
    assert third["another_attempt_made"] is False
    assert third["changes"] == []  # nothing was changed after the last failure


def test_improver_receives_latest_lesson_and_evaluation(agents):
    _, ev, imp = agents
    failing = make_eval((2,))
    ev.side_effect = [failing, make_eval()]
    run()
    lesson, evaluation = imp.await_args.args
    assert lesson.title == "v1"
    assert evaluation == failing


def test_final_state_is_serializable(agents):
    _, ev, _ = agents
    ev.side_effect = [make_eval((1,)), make_eval()]
    state = run()
    json.dumps(state)
    for key in ("lesson", "evaluation", "status", "rejection_log"):
        assert key in state


@pytest.mark.parametrize(
    ("agent", "error", "node_name"),
    [
        ("generate_lesson", InvalidTopicError("bad"), "generate"),
        ("evaluate_lesson", InvalidEvaluationJSONError("bad json"), "evaluate"),
        ("evaluate_lesson", LLMError("boom"), "evaluate"),
        ("improve_lesson", MissingFeedbackError("none"), "improve"),
    ],
)
def test_agent_exceptions_never_report_success(
    agents, monkeypatch, agent, error, node_name
):
    _, ev, _ = agents
    ev.return_value = make_eval((0,))  # so the improver is reached
    monkeypatch.setattr(nodes, agent, AsyncMock(side_effect=error))
    state = run()
    assert state["status"] == "error"
    assert state["error"]["node"] == node_name
    assert state["error"]["error_type"] == type(error).__name__
    assert state["status"] != "passed"


def test_unexpected_bug_is_not_swallowed(agents, monkeypatch):
    monkeypatch.setattr(nodes, "generate_lesson", AsyncMock(side_effect=KeyError("x")))
    with pytest.raises(KeyError):
        run()


def test_default_attempts_come_from_settings(agents, monkeypatch):
    class FakeSettings:
        max_retries = 1

    monkeypatch.setattr(graph_module, "get_settings", lambda: FakeSettings())
    _, ev, imp = agents
    ev.return_value = make_eval((0,))
    state = asyncio.run(run_workflow("RAG"))
    assert state["maximum_attempts"] == 1
    assert imp.await_count == 1


def test_improver_receives_earlier_rounds_as_history(agents):
    _, ev, imp = agents
    ev.side_effect = [make_eval((1,)), make_eval((2,)), make_eval()]
    run()
    first_call, second_call = imp.await_args_list
    assert first_call.kwargs["history"] == []
    history = second_call.kwargs["history"]
    assert len(history) == 1
    assert history[0]["attempt_number"] == 1
    assert history[0]["failed_criteria"] == [CRITERIA[1]]


def test_nodes_pass_topic_requirements_to_evaluator_and_improver(agents):
    from app.evaluation.topics import RAG_REQUIREMENTS

    _, ev, imp = agents
    ev.side_effect = [make_eval((1,)), make_eval()]
    run()
    assert ev.await_args_list[0].kwargs["requirements"] is RAG_REQUIREMENTS
    assert imp.await_args.kwargs["requirements"] is RAG_REQUIREMENTS


def test_topic_without_requirements_passes_none(agents, monkeypatch):
    _, ev, imp = agents
    ev.side_effect = [make_eval((1,)), make_eval()]
    asyncio.run(run_workflow("Photosynthesis", maximum_attempts=2))
    assert ev.await_args_list[0].kwargs["requirements"] is None
    assert imp.await_args.kwargs["requirements"] is None


# --- Phase 3: best-lesson tracking, recheck, richer rejection log -----------------

from app.workflow.nodes import evaluation_score  # noqa: E402


def lesson_versions(agents_tuple, *titles):
    """Make the improver return a different lesson each call."""
    _, _, imp = agents_tuple
    imp.side_effect = [make_lesson(t) for t in titles]


def test_rejected_run_returns_the_best_lesson_not_the_latest(agents):
    gen, ev, _ = agents
    gen.return_value = make_lesson("v1")
    lesson_versions(agents, "v2", "v3")
    # v1: 1 failure (6/7), v2: 3 failures (4/7), v3: 2 failures (5/7)
    ev.side_effect = [make_eval((1,)), make_eval((1, 2, 3)), make_eval((1, 2))]
    state = run()
    assert state["status"] == "rejected"
    assert state["lesson"]["title"] == "v1"
    assert state["best_evaluation_number"] == 1
    assert [c["passed"] for c in state["evaluation"]["checks"]].count(False) == 1
    assert state["latest_lesson"]["title"] == "v3"
    assert [c["passed"] for c in state["latest_evaluation"]["checks"]].count(False) == 2
    assert state["evaluation_count"] == 3  # all three were still evaluated


def test_latest_lesson_is_returned_when_it_is_the_best(agents):
    gen, ev, _ = agents
    gen.return_value = make_lesson("v1")
    lesson_versions(agents, "v2", "v3")
    ev.side_effect = [make_eval((1, 2, 3)), make_eval((1, 2)), make_eval((1,))]
    state = run()
    assert state["status"] == "rejected"
    assert state["lesson"]["title"] == "v3"
    assert state["best_evaluation_number"] == 3
    assert state["latest_lesson"] is None


def test_equal_scores_keep_the_earlier_lesson(agents):
    gen, ev, _ = agents
    gen.return_value = make_lesson("v1")
    lesson_versions(agents, "v2", "v3")
    ev.side_effect = [make_eval((1, 2)), make_eval((3, 4)), make_eval((5, 6))]
    state = run()
    assert state["lesson"]["title"] == "v1"
    assert state["best_evaluation_number"] == 1


def test_passing_lesson_is_the_best(agents):
    gen, ev, _ = agents
    gen.return_value = make_lesson("v1")
    lesson_versions(agents, "v2")
    ev.side_effect = [make_eval((1, 2)), make_eval()]
    state = run()
    assert state["status"] == "passed"
    assert state["lesson"]["title"] == "v2"
    assert state["best_lesson"]["title"] == "v2"
    assert state["best_evaluation_number"] == 2


def test_evaluation_score_ranks_passes_then_covered_concepts():
    def ev(passed, covered):
        return {
            "checks": [{"passed": i < passed} for i in range(7)],
            "concept_checks": [{"covered": i < covered} for i in range(8)],
        }

    assert evaluation_score(ev(6, 0)) > evaluation_score(ev(5, 8))
    assert evaluation_score(ev(5, 6)) > evaluation_score(ev(5, 3))
    assert evaluation_score({"checks": [{"passed": True}]}) == (1, 0)


def test_every_improvement_is_followed_by_a_full_seven_check_evaluation(agents):
    _, ev, imp = agents
    lesson_versions(agents, "v2", "v3")
    ev.side_effect = [make_eval((1,)), make_eval((2,)), make_eval()]
    state = run()
    evaluated_titles = [call.args[0].title for call in ev.await_args_list]
    assert evaluated_titles == ["v1", "v2", "v3"]
    assert imp.await_count == 2
    assert len(state["evaluation"]["checks"]) == 7


def test_rejection_log_records_missing_concepts_and_problem_sentences(agents):
    _, ev, _ = agents
    failing = make_eval((6,)).model_dump()
    failing["concept_checks"] = [
        {
            "concept_id": "augmentation",
            "covered": False,
            "evidence_quote": "",
            "missing_detail": "Not taught.",
        },
        {
            "concept_id": "definition",
            "covered": True,
            "evidence_quote": "Intro",
            "missing_detail": "",
        },
    ]
    failing["checks"][6]["issues"] = [
        {
            "problem": "Thin.",
            "quote": "The exact flagged sentence.",
            "term": "",
            "correction": "Add depth.",
        },
    ]
    ev.side_effect = [EvaluationResult.model_validate(failing), make_eval()]
    state = run()
    entry = state["rejection_log"][0]
    assert entry["missing_concepts"] == ["augmentation"]
    assert entry["problem_sentences"] == ["The exact flagged sentence."]


def test_state_with_best_fields_is_serializable(agents):
    _, ev, _ = agents
    ev.return_value = make_eval((0,))
    json.dumps(run())


# --- Fix 3: every error path returns a consistent lesson/evaluation pair -----------


def assert_paired(state):
    """The returned lesson and evaluation always belong together."""
    if state["best_evaluation"] is not None:
        assert state["lesson"] == state["best_lesson"]
        assert state["evaluation"] == state["best_evaluation"]


def fail_count(evaluation):
    return (
        None
        if evaluation is None
        else [c["passed"] for c in evaluation["checks"]].count(False)
    )


def test_generate_error_has_no_lesson_or_evaluation(agents):
    gen, _, _ = agents
    gen.side_effect = LLMError("boom")
    state = run()
    assert state["status"] == "error"
    assert state["error"]["node"] == "generate"
    assert state["lesson"] is None
    assert state["evaluation"] is None
    assert state["best_lesson"] is None
    assert state["best_evaluation_number"] == 0
    assert state["latest_lesson"] is None


def test_first_evaluation_error_keeps_the_unevaluated_lesson_alone(agents):
    gen, ev, _ = agents
    gen.return_value = make_lesson("v1")
    ev.side_effect = LLMError("boom")
    state = run()
    assert state["error"]["node"] == "evaluate"
    assert state["lesson"]["title"] == "v1"
    assert state["evaluation"] is None  # nothing to pair it with
    assert state["best_evaluation"] is None
    assert state["latest_lesson"] is None


def test_evaluation_error_after_an_improvement_returns_the_best_pair(agents):
    gen, ev, imp = agents
    gen.return_value = make_lesson("v1")
    imp.side_effect = [make_lesson("v2")]
    ev.side_effect = [make_eval((1,)), LLMError("boom")]
    state = run()
    assert state["status"] == "error"
    assert state["error"]["node"] == "evaluate"
    assert_paired(state)
    assert state["lesson"]["title"] == "v1"
    assert fail_count(state["evaluation"]) == 1
    assert state["latest_lesson"]["title"] == "v2"  # improved but never evaluated
    assert state["latest_evaluation"] is None


def test_evaluation_error_returns_a_better_earlier_version_not_the_latest(agents):
    gen, ev, imp = agents
    gen.return_value = make_lesson("v1")
    imp.side_effect = [make_lesson("v2"), make_lesson("v3")]
    ev.side_effect = [make_eval((1, 2)), make_eval((1,)), LLMError("boom")]
    state = run()
    assert_paired(state)
    assert state["lesson"]["title"] == "v2"
    assert fail_count(state["evaluation"]) == 1
    assert state["best_evaluation_number"] == 2
    assert state["latest_lesson"]["title"] == "v3"
    assert state["latest_evaluation"] is None


def test_improve_error_when_the_current_lesson_is_the_best(agents):
    gen, ev, imp = agents
    gen.return_value = make_lesson("v1")
    ev.side_effect = [make_eval((1,))]
    imp.side_effect = LLMError("boom")
    state = run()
    assert state["error"]["node"] == "improve"
    assert_paired(state)
    assert state["lesson"]["title"] == "v1"
    assert fail_count(state["evaluation"]) == 1
    assert state["latest_lesson"] is None


def test_improve_error_when_the_current_lesson_is_worse_than_the_best(agents):
    gen, ev, imp = agents
    gen.return_value = make_lesson("v1")
    imp.side_effect = [make_lesson("v2"), LLMError("boom")]
    ev.side_effect = [make_eval((1,)), make_eval((1, 2, 3))]
    state = run()
    assert state["error"]["node"] == "improve"
    assert_paired(state)
    assert state["lesson"]["title"] == "v1"
    assert fail_count(state["evaluation"]) == 1
    assert state["latest_lesson"]["title"] == "v2"
    assert fail_count(state["latest_evaluation"]) == 3  # v2 WAS evaluated


@pytest.mark.parametrize(
    "scenario", ["generate", "evaluate_first", "evaluate_later", "improve"]
)
def test_error_states_are_serializable_and_never_pass(agents, scenario):
    gen, ev, imp = agents
    if scenario == "generate":
        gen.side_effect = LLMError("boom")
    elif scenario == "evaluate_first":
        ev.side_effect = LLMError("boom")
    elif scenario == "evaluate_later":
        ev.side_effect = [make_eval((1,)), LLMError("boom")]
    else:
        ev.side_effect = [make_eval((1,))]
        imp.side_effect = LLMError("boom")
    state = run()
    json.dumps(state)
    assert state["status"] == "error"
    assert state["error"]["error_type"] == "LLMError"
    assert_paired(state)
