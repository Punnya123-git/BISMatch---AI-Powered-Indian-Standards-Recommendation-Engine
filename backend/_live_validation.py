"""Real end-to-end validation of the pipeline (no fakes anywhere).

Run from the backend directory:
    python _live_validation.py

Everything below the LLM is REAL: the verified BIS dataset, the local ONNX
MiniLM sentence-embedding model, and ChromaDB retrieval. The LLM stage is only
run when a real provider is configured; it is never simulated.
"""

from __future__ import annotations

import argparse
import sys
import time

# The model may echo catalogue text containing characters the default Windows
# console codec cannot encode; never let a print statement kill the report.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.ai.llm.factory import get_llm_provider
from app.core.config import get_settings
from app.rag.pipeline import get_recommendation_pipeline
from app.rag.ranking import rank_standards
from app.schemas.recommendation import RecommendationRequest
from app.services.recommendation_service import RecommendationService
from app.services.standards_service import get_standards_service


QUERIES = [
    ("1) IE3 motor procurement",
     "Supply of 15 kW IE3 three-phase squirrel cage induction motor, "
     "foot mounted, IP55 enclosure, 415 V, 50 Hz"),
    ("2) noise / vibration limits",
     "Permissible noise level limits and vibration measurement for rotating "
     "electrical machines with shaft height 56 mm and above"),
    ("3) unrelated requirement (cement)",
     "Supply of 43 grade ordinary Portland cement in 50 kg moisture-proof "
     "bags with clear marking of the grade and batch number"),
]


def rule(ch: str = "=") -> None:
    print(ch * 78)


USAGE: dict = {}


class _UsageRecorder:
    """Delegates to the real provider and records tokens + finish_reason."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def provider_name(self):
        return self._inner.provider_name

    @property
    def is_configured(self):
        return self._inner.is_configured

    @property
    def model(self):
        return self._inner.model

    def describe(self):
        return self._inner.describe()

    def generate(self, request):
        from app.ai.llm.base import LLMResponse  # noqa: F401

        response = self._inner.generate(request)
        USAGE["prompt_tokens"] = response.usage.get("prompt_tokens")
        USAGE["completion_tokens"] = response.usage.get("completion_tokens")
        USAGE["total_tokens"] = response.usage.get("total_tokens")
        USAGE["finish_reason"] = response.finish_reason
        return response


def main() -> None:
    parser = argparse.ArgumentParser(prog="python _live_validation.py")
    parser.add_argument(
        "--delay",
        type=float,
        default=65.0,
        help=(
            "Seconds to wait between queries. The configured provider has a small "
            "tokens-per-minute budget, so back-to-back queries would 429."
        ),
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help=(
            "Run a single query by its label number (1 = IE3 motor, 2 = noise/"
            "vibration, 3 = cement) instead of all of them."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    provider = get_llm_provider()
    pipeline = get_recommendation_pipeline()
    standards = get_standards_service()
    standards.ensure_loaded()
    readiness = pipeline.readiness()
    service = RecommendationService()
    if provider.is_configured:
        # Record real token usage without changing behaviour: the provider is
        # wrapped, not replaced.
        import app.services.recommendation_service as svc

        svc.get_llm_provider = lambda: _UsageRecorder(provider)

    rule()
    print("PIPELINE CONFIGURATION (real)")
    rule()
    print(f"  dataset            : {settings.standards_dataset_path}")
    print(f"  standards loaded   : {standards.repository.count()}")
    print(f"  embedding provider : {readiness.embedding_provider}")
    print(f"  vector store       : {readiness.vector_store}")
    print(f"  indexed chunks     : {readiness.indexed_chunks}")
    print(f"  retrieval ready    : {readiness.ready}")
    print(f"  llm provider       : {provider.describe()}")
    print(f"  llm is_configured  : {provider.is_configured}")
    if not provider.is_configured:
        print("  llm model          : <none - LLM_MODEL is empty>")
        print("  llm base_url       : <none - LLM_BASE_URL is empty>")

    selected = QUERIES if args.only is None else [QUERIES[args.only - 1]]
    for position, (label, query) in enumerate(selected):
        if position:
            # The provider enforces a tokens-per-minute budget; give it a full
            # window rather than letting the next query fail with HTTP 429.
            time.sleep(args.delay)
        rule()
        print(f"QUERY {label}")
        print(f"  {query}")
        rule()

        # --- A. real retrieved candidates (ONNX embeddings + ChromaDB) -------
        chunks = pipeline.retrieve(query)
        print("A. RETRIEVED CANDIDATES (real ChromaDB similarity)")
        if not chunks:
            print("     (none retrieved)")
        for i, chunk in enumerate(chunks, 1):
            code = (chunk.metadata or {}).get("standard_number") or "(no designation)"
            print(f"     {i:2}. {code:<24} cos={chunk.score:.4f}  chunk={chunk.chunk_id}")

        # --- B. deterministic ranking (unmodified algorithm) ---------------
        outcome = rank_standards(
            chunks,
            requirement=query,
            catalogue_contains=lambda code: standards.repository.get_by_code(code) is not None,
        )
        print("B. DETERMINISTIC CANDIDATE RANKING")
        if not outcome.candidates:
            print("     (no candidate cleared the relevance floor)")
        for cand in outcome.candidates:
            print(f"     rank {cand.rank}. {cand.standard_number:<24} "
                  f"confidence={cand.confidence:.4f}  "
                  f"cos={cand.retrieval_score:.4f}  "
                  f"attr={cand.attribute_score if cand.attribute_score is not None else '-'}")
        print(f"     rejected: below_floor={outcome.below_floor} "
              f"unverified={outcome.unverified}")

        # --- C..I. full service response -------------------------------------
        response = service.analyze(RecommendationRequest(requirement=query))
        ai_ranked = response.pipeline.ranking_stage == "ai_reasoning_ranking"
        print("C. FINAL AI RANKING")
        if ai_ranked:
            for i, rec in enumerate(response.recommendations, 1):
                print(f"     {i}. {rec.code:<24} det_rank={rec.deterministic_rank} "
                      f"conf={rec.confidence:.2f} {rec.applicability_type}")
        else:
            print("     NOT PRODUCED by the AI layer.")
            if not provider.is_configured:
                print("     Reason: no LLM provider is configured.")
            else:
                print("     Reason: the reasoning stage did not yield a trustworthy")
                print("     verdict (see the warnings below). The deterministic")
                print("     ranking is shown as a labelled fallback, not as AI reasoning.")
        print("D. applicability_type : "
              f"{[r.applicability_type for r in response.recommendations] or '-'}")
        print("E. AI reasoning:")
        if ai_ranked:
            for rec in response.recommendations:
                print(f"     {rec.code}: {rec.reasoning[:160]}")
        else:
            print("     NOT PRODUCED")
        print(f"F. evidence used      : {len(response.retrieved_evidence)} retrieved chunk(s)")
        print(f"G. excluded candidates: "
              f"{[w for w in response.warnings if 'excluded' in w] or 'none'}")
        print("H. warnings:")
        for w in response.warnings:
            print(f"     - {w}")
        print(f"I. pipeline status    : status={response.status.value} "
              f"ranking_stage={response.pipeline.ranking_stage} ready={response.pipeline.ready}")
        if USAGE:
            print("LLM usage             : prompt=%s completion=%s total=%s "
                  "finish_reason=%s (TPM limit 8000)"
                  % (USAGE.get("prompt_tokens"), USAGE.get("completion_tokens"),
                     USAGE.get("total_tokens"), USAGE.get("finish_reason")))
        print()


if __name__ == "__main__":
    main()