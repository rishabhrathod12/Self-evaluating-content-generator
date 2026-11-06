"""LangGraph workflow: generate -> evaluate -> (improve -> evaluate)* -> end.

    START -> generate -> evaluate --passed/rejected/error--> END
                            |
                      in_progress (failed, attempts left)
                            v
                         improve -> evaluate ...

The loop is bounded: each pass through ``improve`` consumes one of at most
``maximum_attempts`` (never more than 2) improvement attempts.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.workflow.nodes import evaluate_node, generate_node, improve_node
from app.workflow.state import WorkflowState, initial_state


def route_after_generate(state: WorkflowState) -> Literal["evaluate", "__end__"]:
    """Always evaluate a generated lesson, unless generation failed."""
    return END if state["status"] == "error" else "evaluate"


def route_after_evaluate(state: WorkflowState) -> Literal["improve", "__end__"]:
    """Improve only while the evaluation failed and attempts remain."""
    if state["status"] == "in_progress":
        return "improve"
    return END


def route_after_improve(state: WorkflowState) -> Literal["evaluate", "__end__"]:
    """Always re-evaluate an improved lesson, unless improvement failed."""
    return END if state["status"] == "error" else "evaluate"


def build_graph() -> StateGraph:
    """Build the (uncompiled) workflow graph."""
    graph = StateGraph(WorkflowState)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("improve", improve_node)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", route_after_generate)
    graph.add_conditional_edges("evaluate", route_after_evaluate)
    graph.add_conditional_edges("improve", route_after_improve)
    return graph


workflow = build_graph().compile()


async def run_workflow(
    topic: str, maximum_attempts: int | None = None
) -> WorkflowState:
    """Run the workflow for ``topic`` and return the final state.

    The final state contains the final lesson, final evaluation, status
    (``passed``, ``rejected`` or ``error``) and the rejection log.

    Args:
        topic: The lesson topic.
        maximum_attempts: Improvement attempts allowed (clamped to 0..2). Defaults
            to the ``MAX_RETRIES`` setting.
    """
    if maximum_attempts is None:
        maximum_attempts = get_settings().max_retries
    result = await workflow.ainvoke(initial_state(topic, maximum_attempts))
    return WorkflowState(**result)
