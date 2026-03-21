import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi import status

from app.config import get_settings, Settings
from app.exceptions import UnsupportedFileTypeError, FileTooLargeError, IngestError
from app.api.schemas import IngestResponse
from app.rag.pipeline import ingest


router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    # validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise UnsupportedFileTypeError(ext)

    # validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise FileTooLargeError(settings.max_upload_mb)

    # save to temp file — pipeline expects a file path
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = ingest(tmp_path, original_filename=file.filename)
        result["source"] = file.filename   # use original filename in response

    except Exception as e:
        raise IngestError(str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)   # always clean up temp file

    return IngestResponse(**result)

from app.rag.vectorstore import clear

@router.delete(
    "/collection",
    status_code=status.HTTP_200_OK,
)
async def clear_collection():
    """Wipe the entire vector store. Use with caution."""
    try:
        clear()
        return {"message": "Collection cleared successfully"}
    except Exception as e:
        raise IngestError(str(e))