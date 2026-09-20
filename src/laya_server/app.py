"""The HTTP surface: the Jev evaluation and models endpoints, plus a health check."""

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .engine import MODEL_LIST, to_response
from .schemas import ModelList, SystemOneRequest, SystemOneResponse

log = logging.getLogger("laya_server")

DEFAULT_MAX_QUESTIONS = 128
DEFAULT_MAX_BODY_BYTES = 1 << 20


def _detail(loc: tuple, message: str) -> list[dict]:
    """A validation detail shaped like the one pydantic produces."""
    return [{"type": "value_error", "loc": list(loc), "msg": message}]


def create_app(
    engine: Any,
    *,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    # The documentation routes stay off. This server serves the Jev surface and nothing else.
    app = FastAPI(
        title="laya-server",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def guard_and_log(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds {max_body_bytes} bytes."},
            )
        started = time.perf_counter()
        response = await call_next(request)
        # Content is never logged. Only the route, the outcome and the duration.
        log.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def system_one(body: SystemOneRequest) -> SystemOneResponse:
        if body.model not in engine.names:
            raise HTTPException(
                status_code=404, detail=f"Unknown model {body.model!r}."
            )
        if len(body.questions) > max_questions:
            raise HTTPException(
                status_code=422,
                detail=_detail(
                    ("body", "questions"),
                    f"At most {max_questions} questions per request.",
                ),
            )
        questions = {qid: q.model_dump() for qid, q in body.questions.items()}
        try:
            raw = engine.infer(body.state, questions)
        except ValueError as error:
            # laya rejects a question the schema cannot catch, such as an option set
            # too large for the checkpoint's token window.
            raise HTTPException(
                status_code=422, detail=_detail(("body", "questions"), str(error))
            ) from error
        except Exception:
            log.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed.") from None
        return to_response(raw, engine.model_name)

    @app.get("/v1/models", response_model=ModelList)
    def models() -> ModelList:
        return MODEL_LIST

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model": engine.model_name, "checkpoint": engine.checkpoint}

    return app
