# Self-Evaluating Lesson Content Generator

Generates a beginner-friendly lesson, evaluates it against a strict seven-point
rubric and a topic curriculum, and, when it fails, revises it using the evaluator's
specific feedback and evaluates it again. The assignment topic is
**Introduction to RAG (Retrieval-Augmented Generation)**.

## Status

| Area | State |
| --- | --- |
| Settings, async OpenAI client | Implemented, tested |
| Lesson generator, evaluator, improver | Implemented, tested |
| LangGraph workflow (generate, evaluate, improve, at most 2 improvements) | Implemented, tested |
| Safe debug logging | Implemented, tested |
| HTTP API (`app/main.py`, `app/api/`) | **Not implemented** (placeholder files) |
| Persistent SQLite memory (`app/memory/`) | Implemented, tested (see [Memory](#memory)) |

The workflow is run from Python (see [Running](#running)). Results are returned as a
plain, JSON-serializable dictionary. `app.pipeline` wraps it with SQLite memory (see
[Memory](#memory)); `run_workflow` itself never touches the database.

## How it works

```text
topic -> generate -> evaluate --PASS--------------------------> status "passed"
                        |
                       FAIL, improvement attempts left (max 2)
                        v
                     improve -> evaluate ...
                        |
                       FAIL, no attempts left -----------------> status "rejected"
                                                 (the best-scoring lesson is returned)
```

An agent failure (for example an API error) ends the run with status `error`. An
incomplete lesson is never reported as passed.

### Components

| File | Responsibility |
| --- | --- |
| `app/config.py` | Settings from `.env` and environment variables (Pydantic Settings) |
| `app/llm/client.py` | One shared async OpenAI client (Responses API), structured output |
| `app/agents/generator.py` | Writes the lesson as validated JSON (`LessonContent`) |
| `app/agents/evaluator.py` | Judges the lesson on 7 criteria; code verifies the judgement |
| `app/agents/improver.py` | Revises the lesson from the evaluator's feedback |
| `app/evaluation/topics.py` | The curriculum: concepts, teaching points, workflow, accuracy rules |
| `app/evaluation/rubric.py` | The seven criteria with concrete pass conditions |
| `app/evaluation/schemas.py` | Pydantic models for lessons and evaluations |
| `app/evaluation/concepts.py` | Finds the passage that states each teaching point |
| `app/workflow/` | LangGraph state, nodes and graph (`run_workflow`) |
| `app/debug_log.py` | Optional JSON-lines debug log with credential redaction |

### The evaluation

The seven criteria are accuracy and groundedness, beginner-friendly language,
teaches by example, clear explanations without unexplained jargon, covers key points,
coherent teaching flow, and overall completeness (`app/evaluation/rubric.py`).

For RAG there is also a curriculum of **8 mandatory concepts** (definition, why RAG is
needed, what a language model is, knowledge source and retrieval, augmentation,
generation, the end-to-end workflow with a worked example, limitations), broken into
**12 concrete teaching points**. A concept is `covered` when all its points are stated,
`weak` when some are, and `missing` when none are. The generator is told the
curriculum, the evaluator checks it, and the improver is told exactly which points
are missing.

**The model judges; code verifies.** A small LLM is a noisy judge, so:

- Teaching-point coverage is decided **by code** from the lesson text, which cites the
  passage that states each point. It never depends on the model reproducing text.
- Quotes the model gives for the seven criteria are corrected to the real sentence
  when nearly verbatim; a quote that is not in the lesson is never treated as
  evidence and never ends the run.
- Claims the lesson itself disproves are resolved by code: an "unexplained" term the
  lesson defines (including `X (ACR)` and "also known as"), and an "inaccuracy" whose
  quoted text already says the right thing or negates the claim.
- A quote-less claim that "something is missing" cannot outvote verified evidence for
  *Covers key points* and *Overall completeness*.
- The evaluator runs at temperature 0, and only a malformed response is retried.
- Every decision made by code is recorded in the result's `notes`.

### Result of a run

`run_workflow(topic)` returns a dictionary. The fields that matter:

| Field | Meaning |
| --- | --- |
| `status` | `passed`, `rejected` or `error` |
| `lesson`, `evaluation` | The returned lesson and **its own** evaluation (always the same version) |
| `best_lesson`, `best_evaluation`, `best_evaluation_number` | Highest-scoring evaluated version |
| `latest_lesson`, `latest_evaluation` | Set when a different (later) version exists; `latest_evaluation` is `None` if it was never evaluated |
| `evaluation_count`, `improvement_attempts`, `maximum_attempts` | What happened (the cap is 2) |
| `rejection_log` | One entry per failed evaluation: failed criteria, reasons, missing and weak concepts, exact flagged text, what the following improvement changed, points it left unaddressed, and which concepts the next evaluation resolved |
| `rejection_reason` | Why the run was rejected, in words |
| `error` | The failing node and error, when `status` is `error` |

An empty `changes` on the **last** rejection entry is normal: no improvement follows
the last evaluation. Its `change_summary` says so.

## Setup (Windows)

Requires Python 3.12. Run in PowerShell from the project root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env           # then edit .env
```

### Configuration (`.env`)

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key (never commit it) |
| `OPENAI_MODEL` | Model used by all three agents (default `gpt-4o-mini`) |
| `MAX_RETRIES` | Improvement attempts, 0-2 (default `2`) |
| `APP_ENV` | `development`, `testing` or `production` |
| `DEBUG` | `true` writes model responses, re-asks and validation failures to `outputs/debug_log.jsonl` (default `false`) |
| `DATABASE_URL` | SQLite memory location, `sqlite:///` URLs only (default `sqlite:///./lesson_memory.db`) |

`.env` and everything under `outputs/` are git-ignored.

> **Warning:** with `DEBUG=true`, `outputs/debug_log.jsonl` contains the lesson topic
> and the full text of generated lessons and evaluations. Prompts and API keys are
> never written, and common credential formats are redacted, but the lesson content
> is not. Treat the file as sensitive, do not enable debug logging on a shared
> machine, and delete the log when you are done. It rotates at 5 MB and keeps one
> backup.

## Running

Save this as `run_lesson.py` in the project root and run `python run_lesson.py`
(optionally with a topic: `python run_lesson.py "Introduction to RAG"`). This calls
the OpenAI API and costs a few cents per run.

```python
import asyncio
import json
import os
import sys

from app.llm import close_client
from app.workflow.graph import run_workflow

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "Introduction to RAG"


async def main() -> None:
    try:
        state = await run_workflow(TOPIC)  # uses MAX_RETRIES (default 2)
    finally:
        await close_client()
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/run_result.json", "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    print(
        f"status: {state['status']}  evaluations: {state['evaluation_count']}  "
        f"improvements: {state['improvement_attempts']}"
    )
    if state["rejection_reason"]:
        print("rejected because:", state["rejection_reason"])
    if state["error"]:
        print("error:", state["error"])
    lesson = state["lesson"]
    if lesson:
        print(f"\n# {lesson['title']}\n")
        for number, section in enumerate(lesson["sections"], 1):
            print(f"## {number}. {section['heading']}\n{section['explanation']}")
            print(f"Example: {section['example']}\n")


asyncio.run(main())
```

The full result is saved to `outputs/run_result.json`. A run takes roughly 20 seconds
per model call; a lesson that passes first time needs two calls (generate, evaluate).

## Memory

Finished runs (lessons, evaluations, rejection logs) are stored in a local SQLite
database using only the standard library (`sqlite3`, no extra dependency), and later
runs of the same topic use that history. The evaluator, the seven criteria and the
retry limit (at most 2 improvements) are unchanged.

### End-to-end flow

Run the memory-enabled workflow (calls the OpenAI API, a few cents per run):

```powershell
.\.venv\Scripts\python.exe -m app.pipeline "Introduction to RAG"
```

`app.pipeline.run_with_memory(topic)` does three things in order:

1. **Retrieve.** It reads the newest stored runs of the topic and builds a small
   `memory_context` (`app/memory/context.py`): up to 6 short feedback lines (the
   rejection reason of rejected runs and the evaluator feedback of failed drafts,
   each clipped to about 350 characters, duplicates removed) plus an outline (title
   and section headings only) of the latest lesson that passed.
2. **Run.** It calls the unchanged LangGraph workflow with that context in the state.
   The generator is told to avoid the earlier criticisms and may glance at the
   outline for coverage and ordering, but is told not to copy any earlier lesson.
   The improver gets the feedback lines as extra advice, explicitly subordinate to
   the current evaluation. Lesson text is never passed on.
3. **Save.** It saves the finished run, whether `passed`, `rejected` or `error`.

With no history the context is empty and the agents are called, and prompted,
exactly as before. Rerunning the same topic therefore sees the earlier runs.

**Database errors.** If the memory cannot be opened or read, `MemoryDatabaseError`
is raised before any model call, so nothing is spent. If only the final save fails,
the finished run is not lost: `MemoryRunResult.save_error` holds the message and
`run_id` is `None`. Credentials are scrubbed from error text before it is stored,
and API keys are never part of the state.

From Python:

```python
import asyncio
from app.pipeline import run_with_memory

result = asyncio.run(run_with_memory("Introduction to RAG"))
print(
    result.run_id, result.state["status"], result.memory_context.get("source_run_ids")
)
```

### Storage layer

**Setup.** None beyond `DATABASE_URL` (default `sqlite:///./lesson_memory.db`, a file in
the project root; `sqlite:////absolute/path.db` for an absolute path). The file, its
folder and the tables are created on first use, and opening an existing database is
safe. `*.db` files are git-ignored.

```python
from app.memory import MemoryRepository

memory = MemoryRepository.from_settings()  # or MemoryRepository("path/to.db")
run_id = memory.save_run(
    state
)  # state from run_workflow(...); run_with_memory does this
```

**Schema** (`app/memory/database.py`, `PRAGMA user_version` = 1):

| Table | Holds |
| --- | --- |
| `runs` | One row per run: topic, status (`passed`/`rejected`/`error`), UTC time, attempt counters, rejection reason, error |
| `lessons` | The returned lesson (`role = 'returned'`) and the latest lesson (`'latest'`) as JSON |
| `evaluations` | The evaluation of each stored lesson: overall result, passed checks, covered concepts/points, notes, full JSON |
| `rejection_log` | One row per rejection record, in attempt order |

`save_run` writes a run in a single transaction, so a failure stores nothing. Child
rows cascade when a run is deleted.

**Retrieval.**

| Call | Returns |
| --- | --- |
| `get_run(run_id)` | A `StoredRun` (lessons, evaluations, rejection log) or `None` |
| `list_runs(topic=None, status=None, limit=50)` | Stored runs, newest first |
| `latest_passed_lesson(topic)` | The newest lesson that passed evaluation, as `LessonContent`, or `None` |
| `evaluations_for_topic(topic, only_passed=False)` | Evaluation summaries, newest first |
| `count_runs()`, `delete_run(run_id)` | Housekeeping |

Topics are matched ignoring case and surrounding spaces. Rejected and error runs are
stored for inspection but are never returned by `latest_passed_lesson`.

**Limitations.**

- Only what the final state holds is stored: the returned lesson with its own
  evaluation, the latest lesson (if different), and the rejection log. Intermediate
  versions that were neither best nor latest are not kept.
- History is used only as advice: short feedback lines and one outline, picked by
  recency, not by relevance scoring. A model may still ignore it, and it cannot make a
  rejected topic pass. Nothing is deduplicated (each run adds a row), and failed
  runs' feedback keeps being shown until newer runs push it out of the 5 newest.
- Only `app.pipeline` / `run_with_memory` uses memory. Calling `run_workflow` directly
  stores nothing and uses no history.
- Topics are matched exactly (case-insensitively), not by similarity.
- Local file only: SQLite URLs only, no migrations beyond schema version 1 (a database
  from a newer version is refused), and `:memory:` is not supported because each
  operation opens its own connection.
- Stored lessons and evaluations are plain text and may be sensitive; the database is
  not encrypted.

## Testing

```powershell
pytest                 # 491 tests, all offline: no API calls, no network
ruff check .
ruff format --check .
```

The suite includes **replays of real model output** recorded from live runs
(`tests/fixtures/`), run through the real evaluation code with only the LLM call
mocked, plus end-to-end scenarios with scripted LLM responses.

## Project structure

```text
app/
  config.py            settings
  debug_log.py         safe debug logging
  llm/                 OpenAI client
  agents/              generator, evaluator, improver
  evaluation/          rubric, topics (curriculum), schemas, concept logic
  workflow/            LangGraph state, nodes, graph
  memory/              SQLite storage, retrieval and history context
  pipeline.py          memory-enabled entry point (python -m app.pipeline)
  api/, main.py        placeholders (not implemented)
tests/                 offline tests and real-run fixtures
docs/architecture.md   design notes
outputs/               run results and debug logs (git-ignored)
```

## Known limitations

- **The final code has not been confirmed by a further live run.** It passes all
  offline tests, including replays of every evaluator response recorded in two live
  runs. The most recent live run (before the final fixes) ended in an error, which
  the fixes address; one more live run is the way to confirm.
- **The evaluator is a small LLM** (`gpt-4o-mini` by default). Judgements of the seven
  criteria can vary between lessons. Code verifies and corrects what it can, but it
  cannot make the model perfect. Accuracy rests on the model's own knowledge: there is
  no retrieval or reference corpus, so the evaluator cannot fact-check against sources.
- **Concept coverage is found by word patterns**, calibrated on RAG and on eight real
  recorded lessons. It can credit loosely related text (for example a language-model
  definition that resembles a statement about generation) or miss an unusual wording.
  Only RAG has a curriculum; other topics are judged on the seven generic criteria only.
- **Overrides trade strictness for stability.** For RAG, quote-less claims that
  something is missing cannot outvote verified point evidence. Each such decision is
  in `notes`, but it does shift trust from the model to the code.
- **Genuine inaccuracies can persist.** For example, models like to say "real-time
  information", which RAG does not guarantee. The evaluator flags it, but at most two
  improvements are made, and a rejected run returns the best lesson it produced.
- **No HTTP API** (placeholders only). `requirements.txt`
  also lists packages not used yet (`langchain`, `langchain-openai`, `python-dotenv`,
  `sqlalchemy`, `uvicorn`, `fastapi`).
- **Cost and speed:** each model call takes about 20 seconds; a run makes between 2
  and 8 calls (more only if a response is malformed and has to be retried).
