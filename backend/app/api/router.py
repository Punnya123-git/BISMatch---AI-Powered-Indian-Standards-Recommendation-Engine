"""Aggregates every versioned API router under a single object."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import documents, health, recommendations, standards

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(standards.router)
api_router.include_router(documents.router)
api_router.include_router(recommendations.router)

__all__ = ["api_router"]
