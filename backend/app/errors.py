"""Application-level errors that map to the shared JSON error envelope."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """An expected, client-visible error.

    ``code`` is a stable machine-readable identifier; ``message`` is safe to
    show to users (no stack traces, paths, or secrets).
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
