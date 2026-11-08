"""FastAPI entry point for the Cross-Lingual Hindi RAG Engine."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Cross-Lingual Hindi RAG Engine for Hindi Devanagari Q&A with verifiable citations.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.query import router as query_router
app.include_router(query_router)


@app.get("/")
def read_root():
    """Root endpoint verifying API availability."""
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.GEMINI_MODEL,
    }


@app.get("/health")
@app.get("/api/health")
def healthcheck():
    """Healthcheck endpoint returning operational status."""
    return {"status": "healthy"}
