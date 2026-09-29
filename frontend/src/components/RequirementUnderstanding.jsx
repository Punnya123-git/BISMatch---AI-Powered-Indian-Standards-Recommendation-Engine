/**
 * What the system read out of the requirement text.
 *
 * These are *supporting* signals only: the values are lifted verbatim from the
 * user's own words and are handed to the AI as context. They are shown for
 * transparency, and are deliberately not presented as applicability rules -
 * only the AI analysis may decide that.
 *
 * Nothing is ever invented here. A specification the catalogue could not match is
 * reported as a *catalogue coverage gap*, never as "missing information" the user
 * failed to provide: these values were read straight out of the requirement.
 */
export default function RequirementUnderstanding({ understanding }) {
  const detected = understanding?.detected ?? []
  const unresolved = understanding?.unresolved_families ?? []

  if (!detected.length && !unresolved.length) return null

  return (
    <div className="understanding">
      <p className="understanding__heading">Specifications you provided</p>

      {detected.length ? (
        <ul className="spec-chips">
          {detected.map((item) => (
            <li key={`${item.kind}-${item.surface}`} className="spec-chip">
              <span className="spec-chip__label">{item.label}</span>
              <span className="spec-chip__value">{item.surface}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="understanding__empty">
          No specific technical values were detected. Add details such as power rating, voltage,
          standard reference or material grade for a more precise result.
        </p>
      )}

      {unresolved.length ? (
        <div className="understanding__unresolved">
          <p className="understanding__unresolved-title">Catalogue coverage gaps</p>
          <p className="understanding__unresolved-text">
            These specifications were provided in your requirement, but the currently retrieved
            catalogue records do not state matching values for them, so they could not be used for
            catalogue attribute matching: {unresolved.join(', ')}. This is a limit of the verified
            catalogue records retrieved for this query, not a problem with your requirement.
          </p>
        </div>
      ) : null}

      <p className="understanding__note">{understanding.note}</p>
    </div>
  )
}
