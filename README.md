# Self-Evaluating Lesson Content Generator

## Project Overview

An application that generates beginner-friendly educational lessons, evaluates each lesson against a strict quality rubric, and automatically regenerates lessons that fail, using the evaluator's feedback.

The initial submission topic is **Introduction to RAG (Retrieval-Augmented Generation)**.

> **Status:** Configuration (`app/config.py`) and the async OpenAI client (`app/llm/`) are implemented and tested. Agents, workflow, API logic and database operations are not implemented yet.

## Main Objective

Produce lesson content that reliably meets a defined quality bar by closing the loop: **generate -> evaluate -> improve -> re-evaluate**, with at most two regeneration retries and every attempt persisted.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.12 |
| Orchestration | LangGraph, LangChain |
| LLM | OpenAI Responses API (via the `openai` SDK, `AsyncOpenAI`) |
| API | FastAPI, Uvicorn |
| Validation / settings | Pydantic, pydantic-settings, python-dotenv |
| Persistence | SQLite via SQLAlchemy |
| Quality tooling | Ruff (lint + format), Pytest |

## Planned Architecture

1. **Lesson Generator** drafts a lesson for a topic.
2. **Evaluation Agent** scores it against the rubric and returns a structured PASS/FAIL verdict with feedback.
3. **Improver** regenerates a failed lesson using that feedback (max two retries).
4. **LangGraph** orchestrates the loop as a state graph.
5. **SQLite** stores lessons, evaluations, and attempt history.

See [docs/architecture.md](docs/architecture.md) for details.

## Project Structure

```text
self-evaluating-content-generator/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Environment-based settings
│   ├── agents/              # generator, evaluator, improver
│   ├── workflow/            # LangGraph graph, nodes, state
│   ├── evaluation/          # rubric and Pydantic schemas
│   ├── memory/              # SQLite database and repository
│   └── api/                 # FastAPI routes
├── tests/
├── docs/architecture.md
├── outputs/                 # generated lessons (git-ignored)
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Windows Setup

Requires Python 3.12. Run these in PowerShell from the project root.

### 1. Create the virtual environment

```powershell
py -3.12 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked, run once per session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

For Command Prompt use `.venv\Scripts\activate.bat`.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your own values:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key (never commit it) |
| `OPENAI_MODEL` | Model used for generation and evaluation |
| `DATABASE_URL` | SQLite connection string |
| `MAX_RETRIES` | Regeneration limit, 0-2 (default `2`) |
| `APP_ENV` | `development`, `testing` or `production` (default `development`) |
| `DEBUG` | Debug mode, `true` or `false` (default `false`) |

`.env` is git-ignored; only `.env.example` (placeholders) is tracked.

## Development Tooling

```powershell
ruff check .
ruff format .
pytest
```

## Running the Application

**Upcoming** - the API is not implemented yet. Once available it will be started with:

```powershell
uvicorn app.main:app --reload
```
