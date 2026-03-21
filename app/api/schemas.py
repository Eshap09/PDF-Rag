from pydantic import BaseModel, Field
from typing import Optional


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    source:       str
    chunks_added: int
    replaced:     bool = False
    message:      str = "Document indexed successfully"


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class SourceDocument(BaseModel):
    source:      str
    chunk:       int
    page_number: Optional[int] = None    # None for txt/md files
    extraction:  Optional[str] = None     # "text" or "ocr"
    content:     str



class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceDocument]