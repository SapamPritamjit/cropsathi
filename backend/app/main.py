"""AI4Seva FastAPI application entry point.

Exposes health, crop-information, image-prediction, model-comparison and speech
endpoints with a consistent JSON error envelope and explicit, environment-driven
CORS for browser clients.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .errors import ApiError
from .routers import comparison, crops, health, predict, speech


def _error(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details}},
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return _error(exc.status_code, exc.code, exc.message, exc.details)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return _error(exc.status_code, str(exc.status_code), str(exc.detail))


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error(
        422,
        "validation_error",
        "Request validation failed.",
        jsonable_encoder(exc.errors()),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return _error(500, "internal_error", "An unexpected error occurred.")


def create_app() -> FastAPI:
    """Build the FastAPI application.

    The CORS allow-list is read from ``settings.cors_allowed_origins`` at call
    time so callers (and tests) can build an app for a specific configuration.
    The default is an empty list, which denies every cross-origin browser
    request; no wildcard and no credentials are enabled.
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="AI-powered crop leaf disease detection platform API.",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    application.include_router(health.router)
    application.include_router(crops.router)
    application.include_router(predict.router)
    application.include_router(comparison.router)
    application.include_router(speech.router)

    application.add_exception_handler(ApiError, api_error_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    return application


app = create_app()
