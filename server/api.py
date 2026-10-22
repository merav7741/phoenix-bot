from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from server.guide_catalog import CLAIM_GUIDES
from server.rag import RAGError, get_answer, get_claim_guide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist" / "frontend" / "browser"
MAX_MESSAGE_LENGTH = 2000

app = FastAPI(
    title="Phoenix Insurance Assistant API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank.")
        return value


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: list[ChatMessage] = Field(default_factory=list, max_length=24)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class GuideRequest(BaseModel):
    insurance_type: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("insurance_type", "question")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be blank.")
        return value


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/guide-options")
def guide_options():
    return [{"name": name} for name in CLAIM_GUIDES]


@app.post("/api/chat")
def chat(request: ChatRequest):
    history = [message.model_dump() for message in request.history]
    history.append({"role": "user", "content": request.question})
    try:
        return get_answer(request.question, history=history)
    except RAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/claim-guide")
def claim_guide(request: GuideRequest):
    source_url = CLAIM_GUIDES.get(request.insurance_type)
    if source_url is None:
        raise HTTPException(status_code=422, detail="בחרי תחום ביטוח מהרשימה.")
    try:
        return {
            "chunks": get_claim_guide(source_url, question=request.question),
        }
    except RAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend(path: str):
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Angular build not found. Run `npm run build` in frontend.",
        )
    asset = FRONTEND_DIST / path
    if path and asset.is_file() and FRONTEND_DIST in asset.resolve().parents:
        return FileResponse(asset)
    if path and "." in Path(path).name:
        raise HTTPException(status_code=404, detail="Frontend asset not found.")
    return FileResponse(index_file)
