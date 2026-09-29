"""Verify that the standards index answers plain-language questions.

Usage:

    cd backend
    python -m app.rag.verify_search [--top-k 5] [--json]

Each probe is a question phrased the way an engineer would type it - never a
designation - and is checked against the *live* ChromaDB index built by
``python -m app.rag.index_standards``. A probe passes when the expected BIS
designation is returned within ``--top-k``; the rank and cosine score are shown
so the quality of the retrieval can be judged, not just its presence.

Nothing is simulated: with an empty index or an unconfigured embedding provider
every probe fails and the reason is printed, because fake "hits" would hide the
exact problem this check exists to catch.

Exit codes: 0 all probes passed, 1 at least one probe failed, 2 the index or the
embedding provider is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.rag.pipeline import RecommendationPipeline, get_recommendation_pipeline

#: Plain-language queries and the designation each one should retrieve. Every
#: expectation is a designation that exists in
#: ``data/standards/standards_enriched_bis_verified.json``.
PROBES: tuple[tuple[str, str], ...] = (
    (
        "efficiency classes and performance limits for three-phase a.c. motors",
        "IS 12615:2018",
    ),
    (
        "permissible noise level limits for rotating electrical machines",
        "IS 12065:2025",
    ),
    (
        "measuring mechanical vibration of machines with shaft height 56 mm and higher",
        "IS 12075:2024",
    ),
    (
        "degrees of protection provided by the integral enclosure design IP code",
        "IS/IEC 60034-5:2020",
    ),
    ("designation of methods of cooling for rotating machines", "IS 6362:1995"),
    (
        "dimensions and output series of foot-mounted induction motors frame 56 to 315 L",
        "IS 1231:2019",
    ),
    (
        "single phase a.c. induction motors for general purpose specification",
        "IS 996:2009",
    ),
    (
        "line operated a.c. motors for submersible pump sets",
        "IS 9283:2024",
    ),
    (
        "effect of unbalanced supply voltages on three-phase cage induction motors",
        "IS 13529:2021",
    ),
    ("determining losses and efficiency of machines from tests", "IS/IEC 60034-2-1:2024"),
)


@dataclass(slots=True)
class ProbeResult:
    """Outcome of one semantic search probe."""

    query: str
    expected: str
    rank: int | None = None
    score: float | None = None
    top: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.rank is not None and self.error is None


@dataclass(slots=True)
class VerificationReport:
    """Everything a run of this check produced."""

    embedding_provider: str
    indexed_chunks: int
    top_k: int
    results: list[ProbeResult] = field(default_factory=list)
    unavailable: str | None = None

    @property
    def passed(self) -> bool:
        return self.unavailable is None and all(r.passed for r in self.results)

    @property
    def first_rank_hits(self) -> int:
        """How many probes put the expected designation in position one."""
        return sum(1 for result in self.results if result.rank == 1)


def run_probes(
    *,
    top_k: int = 5,
    probes: tuple[tuple[str, str], ...] = PROBES,
    pipeline: RecommendationPipeline | None = None,
) -> VerificationReport:
    """Run every probe against the live index and report what came back.

    ``pipeline`` may be supplied by callers (such as the opt-in live tests) that
    build their own components; by default the process-wide pipeline configured
    through the environment is used.
    """
    pipeline = pipeline or get_recommendation_pipeline()
    readiness = pipeline.readiness()
    if not readiness.ready:
        return VerificationReport(
            embedding_provider=readiness.embedding_provider,
            indexed_chunks=readiness.indexed_chunks,
            top_k=top_k,
            unavailable="; ".join(readiness.reasons)
            or "the RAG pipeline reported itself as not ready",
        )

    results: list[ProbeResult] = []
    for query, expected in probes:
        try:
            chunks = pipeline.retrieve(query, top_k=top_k)
        except AppError as exc:
            results.append(ProbeResult(query=query, expected=expected, error=str(exc)))
            continue
        designations = [str(chunk.metadata.get("standard_number") or "") for chunk in chunks]
        rank = designations.index(expected) + 1 if expected in designations else None
        results.append(
            ProbeResult(
                query=query,
                expected=expected,
                rank=rank,
                score=chunks[rank - 1].score if rank else None,
                top=designations,
            )
        )
    return VerificationReport(
        embedding_provider=readiness.embedding_provider,
        indexed_chunks=readiness.indexed_chunks,
        top_k=top_k,
        results=results,
    )


def print_report(report: VerificationReport) -> None:
    """Print a human readable pass/fail table for one run."""
    print(f"Embedding provider : {report.embedding_provider}")
    print(f"Indexed chunks     : {report.indexed_chunks}")
    print(f"Top-k per probe    : {report.top_k}")
    print()
    if report.unavailable:
        print(f"NOT VERIFIABLE: {report.unavailable}")
        print("Build the index first: python -m app.rag.index_standards")
        return

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        location = f"rank {result.rank}" if result.rank else "not retrieved"
        score = f", score {result.score:.3f}" if result.score is not None else ""
        print(f"[{status}] {result.expected} <- '{result.query}'")
        print(f"       {location}{score}")
        if result.error:
            print(f"       error: {result.error}")
        elif not result.passed:
            print(f"       returned: {', '.join(result.top) or '(no hits)'}")
    print()
    passed = sum(1 for result in report.results if result.passed)
    print(
        f"{passed}/{len(report.results)} probes retrieved the expected standard "
        f"({report.first_rank_hits} of them as the top hit)."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the verification; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.verify_search",
        description="Check semantic search against the indexed standards.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Hits per probe (default 5).")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    report = run_probes(top_k=args.top_k)
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print_report(report)
    if report.unavailable:
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
