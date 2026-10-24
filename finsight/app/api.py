"""HTTP API.

  GET  /health               liveness + whether the service finished loading
  POST /ask                  {"question": "..."}                         -> answer payload
  POST /verify_chart         {"image_b64", "ticker", "filename"?, "question"?} -> answer payload
  GET  /traces/{request_id}  the stored trace for a request
  GET  /slide?path=...       a slide / uploaded chart image (only files under the data folder)

Run:  uvicorn finsight.app.api:app --port 8000
The service (models, index connections) loads at startup, so the first start takes a while.
A failed request still returns its payload (request_id, trace) with an HTTP error status.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from finsight.app.service import DEFAULT_CHART_QUESTION, IMAGE_SUFFIXES, FinSightService

MAX_IMAGE_BYTES = 8 * 1024 * 1024
STATUS_BY_ERROR = {"BadRequestError": 400, "UnsafeQueryError": 400, "ConfigError": 503, "RetrievalError": 503}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ChartRequest(BaseModel):
    image_b64: str = Field(min_length=8)
    ticker: str = Field(min_length=1, max_length=10)
    filename: str = "chart.png"
    question: str = Field(default=DEFAULT_CHART_QUESTION, max_length=2000)


def _respond(payload: dict) -> JSONResponse:
    status = 200 if not payload.get("error") else STATUS_BY_ERROR.get(payload.get("error_type"), 502)
    return JSONResponse(payload, status_code=status)


def create_app(service: FinSightService | None = None, data_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            from finsight.app.service import build_service

            app.state.service = build_service()
            app.state.service.warm_up()
        yield

    app = FastAPI(title="FinSight", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    def svc() -> FinSightService:
        s = app.state.service
        if s is None:
            raise HTTPException(503, "service is still loading")
        return s

    def data_root() -> Path:
        if data_dir is not None:
            return Path(data_dir).resolve()
        from finsight.core.config import PROJECT_ROOT, get_settings

        d = get_settings().data_dir
        return (d if d.is_absolute() else PROJECT_ROOT / d).resolve()

    @app.get("/health")
    def health():
        return {"status": "ok", "service_ready": app.state.service is not None}

    @app.post("/ask")
    def ask(req: AskRequest):
        return _respond(svc().ask(req.question))

    @app.post("/verify_chart")
    def verify_chart(req: ChartRequest):
        try:
            data = base64.b64decode(req.image_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, "image_b64 is not valid base64")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        return _respond(svc().ask_chart_image(data, req.ticker, req.filename, req.question))

    @app.get("/traces/{request_id}")
    def trace(request_id: str):
        t = svc().tracer.read(request_id)
        if t is None:
            raise HTTPException(404, "no such trace")
        return t

    @app.get("/slide")
    def slide(path: str):
        root = data_root()
        p = Path(path)
        p = (p if p.is_absolute() else root.parent / p).resolve()
        if root not in p.parents or p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            raise HTTPException(404, "no such image")
        return FileResponse(p)

    return app


app = create_app()
