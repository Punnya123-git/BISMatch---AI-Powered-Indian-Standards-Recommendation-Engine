import StatusBadge from './StatusBadge.jsx'

/**
 * The candidate list: what retrieval found, collapsed and clearly marked as
 * *not* a decision.
 *
 * A candidate carries only retrieval signals (rank, similarity, matched terms).
 * It has no confidence, no applicability class and no generated "why it matches"
 * - a model writing prose about a standard it was not deciding on is exactly the
 * false-confidence failure this view must not reintroduce. The disclaimer is
 * rendered server-side, so it travels with the data.
 */
export default function RetrievedCandidates({
  candidates = [],
  recommendations = [],
}) {
  if (!candidates.length) return null

  // Which candidates the AI actually ruled applicable, so the list can show
  // the journey from "retrieved" to "recommended".
  const applied = new Map(
    recommendations.map((item, index) => [item.code, index + 1]),
  )
  const disclaimer =
    candidates[0]?.disclaimer ||
    'Retrieved candidates only. A candidate is not an applicability decision.'

  return (
    <details className="candidates">
      <summary className="candidates__summary">
        <StatusBadge tone="neutral">Potential match</StatusBadge>
        <span className="candidates__title">
          {candidates.length} potential match{candidates.length === 1 ? '' : 'es'} &mdash; not
          recommendations
        </span>
        <span className="candidates__hint">
          Search results only. These are not an applicability decision.
        </span>
      </summary>

      <p className="candidates__disclaimer">
        Potential match — not an applicability decision. Only the AI analysis can determine that a
        standard applies to your requirement.
      </p>

      <ol className="candidates__list">
        {candidates.map((item) => {
          const match = applied.get(item.standard_number)
          const isRecommended = Boolean(match)
          return (
            <li
              key={item.standard_number}
              className={`candidates__item${isRecommended ? ' candidates__item--applied' : ''}`}
            >
              <div className="candidates__item-head">
                <code className="candidates__code">{item.standard_number}</code>
                {isRecommended ? (
                  <StatusBadge tone="success">Recommended #{match}</StatusBadge>
                ) : (
                  <span className="candidates__rank">
                    retrieval rank {item.retrieval_rank}
                  </span>
                )}
              </div>

              <p className="candidates__title-text">{item.title}</p>

              <dl className="candidates__signals">
                {typeof item.similarity_score === 'number' ? (
                  <div>
                    <dt>Similarity</dt>
                    <dd>{item.similarity_score.toFixed(3)}</dd>
                  </div>
                ) : null}
                {item.matched_attributes?.length ? (
                  <div>
                    <dt>Matched terms</dt>
                    <dd>{item.matched_attributes.join(', ')}</dd>
                  </div>
                ) : null}
              </dl>
            </li>
          )
        })}
      </ol>
    </details>
  )
}
