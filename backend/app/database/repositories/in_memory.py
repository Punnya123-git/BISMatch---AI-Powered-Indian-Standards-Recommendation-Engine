"""In-memory standards repository.

Serves as the default implementation until a real dataset loader and/or
PostgreSQL are added. It starts empty by design: the catalogue must be
populated from a genuine, verifiable Indian Standards data source, never from
invented records.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.database.repositories.base import StandardRepository
from app.models.standard import StandardEntity

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def _normalise(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip()).upper()


def _terms(text: str) -> set[str]:
    return {term for term in _WORD_SPLIT.split(text.lower()) if len(term) > 2}


class InMemoryStandardRepository(StandardRepository):
    """Dictionary-backed repository (fine for development and tests)."""

    def __init__(self) -> None:
        self._standards: dict[str, StandardEntity] = {}

    def add_many(self, standards: Sequence[StandardEntity]) -> int:
        for standard in standards:
            self._standards[_normalise(standard.code)] = standard
        return len(standards)

    def get_by_code(self, code: str) -> StandardEntity | None:
        return self._standards.get(_normalise(code))

    def list_all(self, *, limit: int | None = None, offset: int = 0) -> list[StandardEntity]:
        standards = list(self._standards.values())
        standards.sort(key=lambda item: item.code)
        if offset:
            standards = standards[offset:]
        return standards[:limit] if limit is not None else standards

    def keyword_search(self, query: str, *, limit: int = 10) -> list[StandardEntity]:
        query = (query or "").strip()
        if not query:
            return []

        # Exact code matches first, then keyword overlap.
        exact = self._standards.get(_normalise(query))
        query_terms = _terms(query)
        scored: list[tuple[float, StandardEntity]] = []
        for standard in self._standards.values():
            if standard is exact:
                continue
            haystack = " ".join(
                filter(
                    None,
                    [
                        standard.code,
                        standard.title,
                        standard.summary,
                        standard.scope,
                        standard.category,
                        " ".join(standard.keywords),
                        " ".join(
                            str(reference.get("code", ""))
                            for reference in standard.references
                        ),
                        " ".join(
                            str(item)
                            for item in (standard.metadata.get("related_standards") or [])
                        ),
                    ],
                )
            )
            keywords = " ".join(standard.keywords)
            overlap = len(query_terms & _terms(haystack))
            keyword_overlap = len(query_terms & _terms(keywords))
            if overlap or keyword_overlap:
                scored.append((overlap + 2 * keyword_overlap, standard))

        scored.sort(key=lambda item: (-item[0], item[1].code))
        results = [standard for _, standard in scored[:limit]]
        if exact is not None:
            results = [exact, *results][:limit]
        return results

    def count(self) -> int:
        return len(self._standards)

    def clear(self) -> None:
        self._standards.clear()
