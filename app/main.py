"""
TitleForge AI — FastAPI Application
Endpoints:
  GET  /health         — Health check
  GET  /model-info     — Current model metadata
  POST /normalize      — Single title normalization
  POST /bulk-normalize — CSV file upload → normalized CSV
  GET  /               — Serve frontend UI
"""

import io
import os
import time
import logging
import csv
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.model import get_model
from app.schemas import (
    NormalizeRequest,
    NormalizeResponse,
    BulkNormalizeResponse,
    HealthResponse,
    ModelInfoResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger("titleforge")

# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 TitleForge AI starting up...")
    get_model()  # Pre-warm model on startup
    logger.info("✅ Model ready. TitleForge AI is live!")
    yield
    logger.info("🛑 TitleForge AI shutting down.")


app = FastAPI(
    title="TitleForge AI",
    description=(
        "A product title normalization engine that converts raw, inconsistent "
        "vendor titles into clean, standardized catalog titles using a fine-tuned LLM."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"], summary="Redirect to UI")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Returns service health status and model readiness."""
    model = get_model()
    return HealthResponse(
        status="ok",
        model_backend=model.info["model_backend"],
        model_loaded=model.is_ready(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["General"])
async def model_info():
    """Returns current model metadata and configuration."""
    return ModelInfoResponse(**get_model().info)


@app.post("/normalize", response_model=NormalizeResponse, tags=["Normalization"])
async def normalize(request: NormalizeRequest):
    """
    Normalize a single raw product title.

    - **raw_title**: The messy vendor title to clean up.
    - Returns: cleaned, standardized title.
    """
    model = get_model()
    t0 = time.perf_counter()
    try:
        normalized = model.normalize(request.raw_title)
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Normalized in {elapsed_ms:.1f}ms  |  '{request.raw_title}' → '{normalized}'")

    return NormalizeResponse(
        raw_title=request.raw_title,
        normalized_title=normalized,
        model_backend=model.info["model_backend"],
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/bulk-normalize", tags=["Normalization"])
async def bulk_normalize(file: UploadFile = File(...)):
    """
    Normalize a batch of raw titles from a CSV file.

    - **file**: CSV file with a `raw_title` column.
    - Returns: Downloadable CSV with added `normalized_title` column.

    Accepted CSV formats:
    - Must have a `raw_title` column (or first column will be used as fallback)
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    # Flexible column detection
    if "raw_title" not in df.columns:
        if len(df.columns) >= 1:
            df = df.rename(columns={df.columns[0]: "raw_title"})
        else:
            raise HTTPException(status_code=400, detail="CSV must have at least one column for raw titles.")

    model = get_model()
    t0 = time.perf_counter()
    results, succeeded, failed = [], 0, 0

    for _, row in df.iterrows():
        raw = str(row["raw_title"]).strip()
        try:
            normalized = model.normalize(raw)
            results.append({"raw_title": raw, "normalized_title": normalized, "status": "ok"})
            succeeded += 1
        except Exception as e:
            results.append({"raw_title": raw, "normalized_title": "", "status": f"error: {str(e)}"})
            failed += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Bulk: {succeeded} ok, {failed} failed in {elapsed_ms:.1f}ms")

    # Build output CSV
    output_df = pd.DataFrame(results)
    stream = io.StringIO()
    output_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=normalized_{file.filename}"},
    )


# ── Dev server entry ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
