# Architecture

> Planned design. Nothing here is implemented yet.

## Flow

```text
Topic -> Generator -> Evaluator --PASS--> Save & finish
                          |
                         FAIL (retries < 2)
                          v
                      Improver -> Evaluator ...
                          |
                         FAIL (retries exhausted) -> Save last attempt as FAIL
```

## Components

### Lesson Generator
Produces a beginner-friendly lesson for a topic (initially *Introduction to RAG*) using an OpenAI model, returning structured output.

### Evaluation Agent
Scores the lesson against a strict rubric (`app/evaluation/rubric.py`) and returns a Pydantic-validated result (`app/evaluation/schemas.py`): per-criterion results, an overall verdict, and actionable feedback.

### PASS/FAIL Evaluation Approach
Each rubric criterion is judged PASS or FAIL with a short justification. A lesson is **PASS only if every criterion passes**; any failing criterion makes the overall verdict **FAIL**. Binary verdicts keep the routing decision deterministic and auditable.

### Feedback and Improvement
On FAIL, the Improver receives the original lesson plus the evaluator's feedback and produces a revised lesson that targets the failed criteria. The revision is re-evaluated.

### LangGraph Orchestration
The loop is a LangGraph state graph (`app/workflow/`): shared state (`state.py`), nodes for generate/evaluate/improve/persist (`nodes.py`), and a conditional edge after evaluation that routes to the improver, or to the end (`graph.py`).

### Maximum Two Regeneration Retries
After the initial generation, at most **two** regenerations are allowed (up to three attempts total). The limit is configurable via `MAX_REGENERATION_RETRIES` (default `2`). If the lesson still fails, the final attempt is stored with a FAIL verdict rather than looping indefinitely.

### SQLite Persistent Memory
SQLite (via SQLAlchemy, `app/memory/`) persists every lesson attempt, its evaluation, and feedback, so history survives restarts and can be inspected or reused. Database files are git-ignored.
