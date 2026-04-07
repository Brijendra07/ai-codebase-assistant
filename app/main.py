from fastapi import FastAPI

from app.api.routes_chunking import router as chunking_router
from app.api.routes_embeddings import router as embeddings_router
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.core.config import settings
from app.core.logger import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend-first AI Codebase Assistant for repository understanding.",
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(chunking_router)
app.include_router(embeddings_router)
app.include_router(query_router)


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    logger.info("Root endpoint called")
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
