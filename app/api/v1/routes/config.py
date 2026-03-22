import os
from pathlib import Path

from fastapi import APIRouter, status
from pydantic import BaseModel

router   = APIRouter()
RAG_HOME = Path.home() / ".rag"


class ConfigRequest(BaseModel):
    groq_api_key: str


class ConfigResponse(BaseModel):
    message: str
    key_set: bool


@router.post("/config", response_model=ConfigResponse, status_code=status.HTTP_200_OK)
async def set_config(request: ConfigRequest):
    """Save GROQ_API_KEY to ~/.rag/.env and apply immediately."""
    key = request.groq_api_key.strip()

    if not key:
        return ConfigResponse(message="No key provided", key_set=False)

    RAG_HOME.mkdir(parents=True, exist_ok=True)
    env_file = RAG_HOME / ".env"
    env_file.write_text(f"GROQ_API_KEY={key}\n")

    # apply to current process immediately
    os.environ["GROQ_API_KEY"] = key

    return ConfigResponse(message="API key saved successfully", key_set=True)


@router.get("/config", response_model=ConfigResponse, status_code=status.HTTP_200_OK)
async def get_config():
    """Check if GROQ_API_KEY is configured."""
    key = os.environ.get("GROQ_API_KEY", "")

    if not key:
        env_file = RAG_HOME / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()

    return ConfigResponse(
        message="Key configured" if key else "No key set",
        key_set=bool(key),
    )