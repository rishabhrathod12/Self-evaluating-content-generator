# Architecture

Design notes for the self-evaluating lesson generator. See the README for setup and
usage. This describes what is implemented.

## Flow

```text
topic -> generate -> evaluate --PASS--> passed
                        |
                       FAIL (improvements < MAX_RETRIES, at most 2)
                        v
                     improve -> evaluate ...
                        |
                       FAIL (limit reached) -> rejected, best lesson returned
```

A LangGraph state graph (`app/workflow/graph.py`) with three nodes and conditional
edges. The loop is bounded: each pass through `improve` uses one of at most two
improvement attempts, so the workflow always terminates.

## Agents

- **Generator** (`app/agents/generator.py`): builds a prompt from the topic, the
  curriculum, the workflow steps and the accuracy guardrails, calls the model, and
  validates the JSON as `LessonContent` (strict Pydantic).
- **Evaluator** (`app/agents/evaluator.py`): judges the lesson on seven criteria with
  structured output and temperature 0, then verifies the judgement in code (below).
- **Improver** (`app/agents/improver.py`): receives the lesson, the full evaluation,
  the earlier rounds and the exact missing teaching points. A revision must change the
  lesson, preserve content that had no reported problem, and (checked by code) state
  each requested point. If a point is still missing after one re-ask, the revision is
  returned and the gap is recorded: a revision is never reported as a success.

All three reuse one async OpenAI client (`app/llm/client.py`, Responses API).

## The curriculum and evaluation

`app/evaluation/topics.py` defines, for RAG, eight concepts made of twelve teaching
points. Each point has *anchor groups*: lenient synonym groups used only as a sanity
check on text, never as the quality judgement.

Design decision: **the model judges, code verifies and cites.**

1. The model returns the seven criterion verdicts and, per teaching point, whether it
   is covered (no quotes for points).
2. Criterion quotes are corrected to the exact sentence when 90% similar or better;
   otherwise they are never evidence (replaced, kept as an unquoted claim, or the
   unverifiable issue is dropped). Nothing here ends the run.
3. Claims the lesson disproves are resolved: terms it defines (`X is`, `called X`,
   `X (ACR)`, "shortened to", "also known as"), accuracy issues whose quote already
   states the correction or negates the accused claim.
4. For each teaching point, code searches the lesson body for the passage that best
   states it. Found means covered (that passage is the evidence); not found means not
   covered, whatever the model said.
5. Concept status is derived: all points stated is `covered`, some `weak`, none
   `missing`. A weak or missing concept fails *Overall completeness* with the exact
   points still to state.
6. Quote-less "missing" claims for *Covers key points* and *Overall completeness* are
   reconciled with the verified point evidence (they cannot outvote it).
7. Feedback is rebuilt from the remaining failures: reason, flagged text, correction,
   fix.

Every decision made by code is appended to the evaluation's `notes`.

Why: in live runs the small model copied requirement text as quotes, flipped verdicts
between near-identical lessons and repeated vague absence claims. Reproducing text
and judging presence are exactly what code does reliably.

## Workflow state

`WorkflowState` (`app/workflow/state.py`) is a plain, JSON-serializable dictionary.
After every evaluation the best lesson (most passed checks, then covered concepts,
then covered points; the earlier one wins ties) is tracked together with its own
evaluation. A rejected or errored run returns that best pair as `lesson` and
`evaluation`, and keeps a different latest version in `latest_lesson` /
`latest_evaluation` (`None` if it was never evaluated).

The rejection log has one entry per failed evaluation: failed criteria, reasons,
missing and weak concepts, exact flagged text, what the improvement that followed
changed, points it left unaddressed, and which concepts the next evaluation resolved.
`rejection_reason` states in words why the run was rejected.

## Debug logging

`app/debug_log.py` (opt-in with `DEBUG=true`) appends JSON lines with the model
responses, re-asks and validation failures. Prompts and API keys are never logged,
common credential formats and secret-named settings are redacted, the file rotates at
5 MB, and a logging failure can never interrupt a run. The log can contain lesson
content, so it is git-ignored and should be treated as sensitive.

## Memory

`app/pipeline.py` (`run_with_memory`, or `python -m app.pipeline "<topic>"`) is the
application entry point. The graph itself is unchanged and never touches the database:

```text
topic -> MemoryRepository.list_runs / latest_passed_lesson
      -> build_memory_context  (<= 6 clipped feedback lines + outline of last pass)
      -> run_workflow(topic, memory_context=...)   generate -> evaluate -> improve*
      -> MemoryRepository.save_run(final state)    passed | rejected | error
```

The context lives in the workflow state (`memory_context`, `{}` when there is no
history). `generate_node` and `improve_node` pass it to the agents only when it is
non-empty, so without history the calls and prompts are identical to before. The
generator prompt adds the earlier criticisms and a loose outline reference with an
instruction not to copy; the improver prompt adds the feedback lines as advice
subordinate to the current evaluation. Evaluator scoring, the seven criteria and the
retry limit (2) are not involved.

Failure handling: a memory that cannot be opened or read raises `MemoryDatabaseError`
before any model call; a failed final save is reported in the result
(`save_error`) while the finished run is still returned. Free text that is stored
(error messages, rejection reason) is redacted for credentials first. Storage
details and limitations are in the README's Memory section.

## Not implemented

An HTTP API (`app/api/`, `app/main.py` are placeholders).
