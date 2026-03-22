from fastapi import APIRouter

from app.api.v1.routes.ingest import router as ingest_router
from app.api.v1.routes.query  import router as query_router
from app.api.v1.routes.config import router as config_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(ingest_router, tags=["ingest"])
v1_router.include_router(query_router,  tags=["query"])
v1_router.include_router(config_router, tags=["config"])