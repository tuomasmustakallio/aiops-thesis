"""FastAPI backend for CI/CD Failure Prediction thesis experiment."""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="CI/CD Thesis Backend")

# Path to frontend build directory (populated by CI)
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/health")
def health_check():
    """Health check endpoint for monitoring and deployment verification."""
    return {"status": "healthy", "service": "thesis-backend"}


# Serve frontend static files if the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React frontend for all non-API routes."""
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fallback to index.html for SPA routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"error": "Frontend not built"}
