const COPY = {
  idle: {
    title: 'Ready when you are',
    body: 'Describe a procurement or product requirement — or attach a tender document — and press Analyze Requirement. The engine retrieves matching Indian Standards and explains each one against the evidence it was judged on.',
  },
  noRequirement: {
    title: 'No requirement entered',
    body: 'Enter a requirement above before running the analysis. A few words are enough: e.g. "15 kW IE3 three-phase squirrel cage induction motor, 415 V, 50 Hz, IP55".',
  },
  noStandards: {
    title: 'No applicable standard identified',
    body: 'Nothing in the verified catalogue was judged to apply to this requirement. The catalogue currently covers a limited verified subset of Indian Standards for rotating electrical machines, so requirements outside that scope will legitimately return nothing. Broaden the requirement with more technical detail, or try one of the examples.',
  },
  // --- coverage-aware no-result messaging --------------------------------
  // The four "nothing here" outcomes are genuinely different, and collapsing
  // them would tell a user that our catalogue has no answer when the truth is
  // that the provider failed, or that the AI considered the candidates and
  // rejected them. `no_result_reason` is stated by the backend, never guessed.
  aiUnavailable: {
    title: 'AI applicability analysis unavailable',
    body: 'We found potentially relevant catalogue entries, but we could not safely determine which standards apply. Nothing is recommended without a completed AI applicability analysis.',
  },
  aiRuledNone: {
    title: 'No applicable standard identified',
    body: 'AI reviewed the available verified candidates and found no applicable standard. The candidates were evaluated and ruled out — this is a completed analysis, not a failure.',
  },
  noCandidates: {
    title: 'Insufficient verified coverage',
    body: 'The current verified knowledge base does not contain enough information to make a recommendation for this requirement. This is a limit of this assistant’s local index — it is not a statement that no Indian Standard exists.',
  },
  offline: {
    title: 'Backend unavailable',
    body: 'The health check did not reach the API, so no analysis can run. Start the FastAPI server and use Recheck in the header; nothing is displayed from a cache because no cached recommendation data is stored.',
  },
}

const COPY_ICON = {
  offline: '!',
  noStandards: '?',
  noCandidates: '?',
  aiUnavailable: '!',
  aiRuledNone: '—',
}

/**
 * Polished placeholder for every non-result situation, so the page is never a
 * blank screen. `variant` picks the copy; callers pass the backend's own message
 * through `detail` when there is one.
 */
export default function EmptyState({ variant = 'idle', detail, action }) {
  const copy = COPY[variant] || COPY.idle

  return (
    <div className={`card state-card state-card--${variant}`}>
      <div className="state-card__icon" aria-hidden="true">
        {COPY_ICON[variant] || '○'}
      </div>
      <div className="state-card__body">
        <h2 className="state-card__title">{copy.title}</h2>
        <p className="state-card__text">{copy.body}</p>
        {detail ? <p className="state-card__meta">{detail}</p> : null}
        {action}
      </div>
    </div>
  )
}