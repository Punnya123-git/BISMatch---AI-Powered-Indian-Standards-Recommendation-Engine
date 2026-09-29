import { summarizeResult } from '../utils/pipelineState.js'
import EvidencePanel from './EvidencePanel.jsx'
import RecommendationCard from './RecommendationCard.jsx'
import RetrievedCandidates from './RetrievedCandidates.jsx'
import ResultsSummary from './ResultsSummary.jsx'
import WarningsPanel from './WarningsPanel.jsx'

/**
 * Pick the empty-state variant from the backend's own `no_result_reason`.
 *
 * The backend states *why* the list is empty; we never infer it. Falling back to
 * candidate counts is a last-resort default for older responses, not the rule.
 *
 * `retrieval_unavailable` is deliberately NOT folded into `aiUnavailable`: when
 * search itself could not run, the AI stage never executed, so telling the user
 * "AI analysis unavailable" would blame a stage that never started.
 */
function noResultVariant(result) {
  switch (result.no_result_reason) {
    case 'ai_ruled_none':
      return 'aiRuledNone'
    case 'insufficient_coverage':
      return 'noCandidates'
    case 'ai_unavailable':
      return 'aiUnavailable'
    case 'retrieval_unavailable':
      return 'retrievalUnavailable'
    default: {
      const candidates = result.retrieved_candidates || []
      if (!candidates.length) return 'noCandidates'
      return result.ai_reasoning?.status === 'completed' ? 'aiRuledNone' : 'aiUnavailable'
    }
  }
}

/**
 * Renders a finished `RecommendationResponse`.
 *
 * The central rule this view enforces: recommendation cards exist only when the
 * backend returned a non-empty `recommendations` list, which it does only after a
 * grounded AI verdict. When that list is empty the section shows an explanation
 * instead of cards - retrieved candidates move to a collapsed, explicitly
 * disclaimed panel so evidence stays inspectable without ever being dressed up
 * as an answer.
 */
/**
 * Copy for every "no applicable standards" outcome.
 *
 * The four states are genuinely different facts reported by the backend and must
 * never be collapsed: saying "AI unavailable" when search never ran, or
 * "insufficient coverage" when the AI ruled, would both mislead a user about
 * whether BISMatch actually reached a verdict.
 */
const NO_RESULT_COPY = {
  aiUnavailable: {
    icon: '!',
    title: 'AI applicability analysis unavailable',
    body: 'Potentially relevant catalogue entries were found, but the AI applicability analysis could not be completed, so no standard is presented as applicable. The potential matches below are search results only.',
  },
  retrievalUnavailable: {
    icon: '!',
    title: 'Standards search unavailable',
    body: 'The search over the verified knowledge base could not run for this request, so no evidence was retrieved and no applicability analysis was attempted. This is a retrieval problem, not a statement about BIS coverage.',
  },
  aiRuledNone: {
    icon: '—',
    title: 'No applicable standard identified',
    body: 'AI reviewed the available verified candidates and found no applicable standard.',
  },
  noCandidates: {
    icon: '?',
    title: 'Insufficient verified coverage',
    body: 'The current verified knowledge base does not contain enough information to make a recommendation for this requirement.',
  },
}

/**
 * Renders a finished `RecommendationResponse`.
 *
 * The central rule this view enforces: recommendation cards exist only when the
 * backend returned a non-empty `recommendations` list, which it does only after a
 * grounded AI verdict. When that list is empty the section shows an explanation
 * instead of cards - retrieved candidates move to a collapsed, explicitly
 * disclaimed panel so evidence stays inspectable without ever being dressed up
 * as an answer.
 */
export default function ResultsSection({ result, onRetry, onRefine }) {
  if (!result) return null

  const recommendations = result.recommendations || []
  const candidates = result.retrieved_candidates || []
  const { aiRanked } = summarizeResult(result)
  const noResult = noResultVariant(result)
  const noResultCopy = NO_RESULT_COPY[noResult] || NO_RESULT_COPY.noCandidates

  return (
    <div className="results">
      <ResultsSummary result={result} />

      <WarningsPanel warnings={result.warnings} />

      {/* Part 4: when the AI did not complete, the heading must never read
          "Applicable standards" - there are none, and implying otherwise is the
          exact failure this product must not ship. */}
      {recommendations.length ? (
        <section className="card recs-card" aria-label="Applicable standards">
          <header className="card__header">
            <div>
              <p className="card__eyebrow">Final result</p>
              <h2 className="card__title">Applicable Indian Standards</h2>
            </div>
            <span className="count-chip">{recommendations.length} identified</span>
          </header>

          <p className="recs-card__legend">
            Ranked by AI applicability analysis. &ldquo;Retrieved candidate&rdquo; shows where each
            standard sat after semantic search, so you can see where the AI changed the initial
            order.
          </p>

          <div className="rec-list">
            {recommendations.map((recommendation, index) => (
              <RecommendationCard
                key={recommendation.code || index}
                recommendation={recommendation}
                rank={index + 1}
                aiRanked={aiRanked}
                defaultOpen={index === 0}
              />
            ))}
          </div>
        </section>
      ) : null}

      {!recommendations.length ? (
        <section className="card recs-card" aria-label="No applicable standards">
          <div className={`state-card state-card--${noResult}`}>
            <div className="state-card__icon" aria-hidden="true">
              {noResultCopy.icon}
            </div>
            <div className="state-card__body">
              <h2 className="state-card__title">{noResultCopy.title}</h2>
              <p className="state-card__text">{noResultCopy.body}</p>
              {result.coverage_note ? (
                <p className="state-card__meta">{result.coverage_note}</p>
              ) : null}

              <div className="state-card__actions">
                {noResult !== 'noCandidates' && onRetry ? (
                  <button type="button" className="primary-button" onClick={onRetry}>
                    Retry analysis
                  </button>
                ) : null}
                {onRefine ? (
                  <button type="button" className="ghost-button" onClick={onRefine}>
                    Refine requirement
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {/* Candidates are always collapsed and always disclaimed: they are the
          retrieval layer's output, not the AI's conclusion. */}
      <RetrievedCandidates candidates={candidates} recommendations={recommendations} />

      <EvidencePanel items={result.retrieved_evidence} />
    </div>
  )
}