"""Shared response/request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error payload returned by every failing endpoint."""

    code: str = Field(description="Machine readable error code.")
    message: str = Field(description="Human readable explanation.")
    details: Any | None = Field(default=None, description="Optional extra context.")


class MessageResponse(BaseModel):
    """Simple acknowledgement payload."""

    message: str
