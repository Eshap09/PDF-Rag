from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.config import get_settings
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs on startup — warm up the embedding model so the first
    # request isn't slow waiting for the model to load
    print("[startup] warming up embedding model...")
    from app.rag.embedder import get_embedder
    get_embedder()
    print("[startup] ready")

    yield  # app runs here

    # runs on shutdown
    print("[shutdown] cleaning up...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG API",
        description="Retrieval Augmented Generation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allows browser clients to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # tighten this in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # register v1 routes — all endpoints will be under /api/v1/...
    app.include_router(v1_router, prefix="/api")

    app.mount("/ui", StaticFiles(directory="app/ui"), name="ui")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse("app/ui/index.html")


    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()