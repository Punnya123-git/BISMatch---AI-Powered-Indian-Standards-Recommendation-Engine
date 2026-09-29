/**
 * Sticky product header.
 *
 * Deliberately free of infrastructure detail. A procurement user needs to know
 * the service is reachable, not which embedding model or vector store is behind
 * it - that lives in the collapsed "Technical details" panel instead.
 */
export default function Header({ status, error, onRefresh }) {
  const tone = status === 'online' ? 'success' : status === 'loading' ? 'neutral' : 'error'
  const label =
    status === 'online'
      ? 'Service online'
      : status === 'loading'
        ? 'Connecting'
        : 'Service offline'

  return (
    <header className="topbar">
      <div className="topbar__inner">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            BM
          </span>
          <div className="brand__text">
            <p className="brand__name">BISMatch</p>
            <p className="brand__tagline">AI-Powered Indian Standards Recommendation Engine</p>
          </div>
        </div>

        <div className="topbar__status">
          <div className={`status-pill status-pill--${tone}`} role="status" aria-live="polite">
            <span className="status-pill__dot" aria-hidden="true" />
            <span className="status-pill__label">{label}</span>
            <button
              type="button"
              className="status-pill__refresh"
              onClick={onRefresh}
              disabled={status === 'loading'}
            >
              {status === 'loading' ? 'Checking…' : 'Recheck'}
            </button>
          </div>

          {error ? <p className="topbar__error">{error}</p> : null}
        </div>
      </div>
    </header>
  )
}