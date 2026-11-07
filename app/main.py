"""Local web UI: ``python -m uvicorn app.main:app`` then open http://127.0.0.1:8000"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.llm import close_client
from app.memory.database import MemoryDatabaseError
from app.pipeline import run_with_memory

app = FastAPI(title="Self-Evaluating Lesson Generator")
_PAGE = Path(__file__).parent / "web" / "index.html"


class GenerateRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE.read_text(encoding="utf-8")


@app.post("/api/generate")
async def generate(body: GenerateRequest) -> JSONResponse:
    try:
        result = await run_with_memory(body.topic.strip())
    except MemoryDatabaseError as exc:
        return JSONResponse({"error": f"Memory error: {exc}"}, status_code=500)
    except Exception as exc:  # surface config/API problems in the UI
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    finally:
        await close_client()
    return JSONResponse(
        {
            "run_id": result.run_id,
            "save_error": result.save_error,
            "history_runs": result.memory_context.get("source_run_ids", []),
            "state": result.state,
        }
    )
