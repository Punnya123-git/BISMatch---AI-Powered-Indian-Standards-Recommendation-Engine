import {
  RANKING_STAGE,
  RANKING_STAGE_HINTS,
  RANKING_STAGE_LABELS,
  STATUS_LABELS,
  STATUS_TONES,
} from '../utils/constants.js'
import { pluralize } from '../utils/formatters.js'
import { summarizeResult } from '../utils/pipelineState.js'
import RequirementUnderstanding from './RequirementUnderstanding.jsx'
import StatusBadge from './StatusBadge.jsx'

/**
 * Headline results header.
 *
 * Every number and label here is read straight off the response: the count from
 * `recommendations.length`, the stage from `pipeline.ranking_stage` and the
 * candidate count from `retrieved_candidates`. Nothing is inferred, and no
 * number is shown for a stage that did not run.
 */
export default function ResultsSummary({ result }) {
  if (!result) return null

  const { count, aiRanked, rankingStage, consideredCount } = summarizeResult(result)
  const status = result.status
  const tone = STATUS_TONES[status] || 'neutral'
  const statusLabel = STATUS_LABELS[status] || status
  const aiStatus = result.ai_reasoning?.status ?? null
  const aiRuledNothing = !aiRanked && aiStatus === 'completed'

  // The stage name alone is misleading here: `ai_reasoning_unavailable` is also
  // returned when the AI *did* complete and selected nothing. Reporting that as
  // "unavailable" would tell a user the model failed when it actually ruled.
  const stageLabel = aiRanked
    ? (RANKING_STAGE_LABELS[rankingStage] || rankingStage)
    : aiRuledNothing
      ? 'Complete — no applicable standard found'
      : RANKING_STAGE_LABELS[rankingStage] || rankingStage || 'Not reported'

  return (
    <section className="card summary-card" aria-label="Results summary">
      <header className="summary-card__head">
        <div>
          <p className="card__eyebrow">Results</p>
          <h2 className="summary-card__headline">
            {count > 0
              ? `${pluralize(count, 'applicable standard')} identified`
              : 'No applicable standard identified'}
          </h2>
        </div>
        <StatusBadge tone={tone} title={result.message}>
          {statusLabel}
        </StatusBadge>
      </header>

      <p className="summary-card__message">{result.message}</p>

      <dl className="stat-grid">
        <div className="stat-grid__item">
          <dt>Applicable standards</dt>
          <dd>{count}</dd>
        </div>
        <div className="stat-grid__item">
          <dt>Standards evaluated</dt>
          <dd>{consideredCount}</dd>
          <dd className="stat-grid__hint">potential matches reviewed</dd>
        </div>
        <div className="stat-grid__item">
          <dt>Analysis status</dt>
          <dd className="stat-grid__value--text">{stageLabel}</dd>
        </div>
      </dl>

      {/* The stage is never described as a fallback *result*: on this API an
          empty list is always a deliberate "no applicability decision", never a
          degraded answer dressed up as a recommendation. */}
      <div
        className={`reasoning-banner reasoning-banner--${
          aiRanked ? 'ai' : aiRuledNothing ? 'neutral' : 'unavailable'
        }`}
      >
        <StatusBadge tone={aiRanked ? 'info' : aiRuledNothing ? 'neutral' : 'warning'}>
          {aiRanked
            ? 'AI applicability analysis'
            : aiRuledNothing
              ? 'AI found no applicable standard'
              : 'AI analysis unavailable'}
        </StatusBadge>
        <p className="reasoning-banner__text">
          {aiRanked
            ? RANKING_STAGE_HINTS[RANKING_STAGE.AI]
            : aiRuledNothing
              ? 'The AI reviewed every standard found in the catalogue and determined that none applies to this requirement. This is a completed analysis, not a failure.'
              : rankingStage
                ? RANKING_STAGE_HINTS[rankingStage] || ''
                : 'The analysis status was not reported for this response.'}
        </p>
      </div>

      {count === 0 && consideredCount > 0 && !aiRanked ? (
        <p className="summary-card__note">
          {aiRuledNothing
            ? 'The AI examined the standards found in the catalogue and judged that none of them applies to this requirement. No standard is listed as applicable.'
            : 'Because no AI applicability analysis was completed, no standard is listed as applicable. The potential matches below are search results only.'}
        </p>
      ) : null}

      <RequirementUnderstanding understanding={result.requirement_understanding} />
    </section>
  )
}