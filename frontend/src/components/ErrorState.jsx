const HINTS = {
  network_error: 'The browser could not reach the API. Check that the backend is running and that the Vite proxy (or VITE_API_BASE_URL) points at it.',
  validation_error: 'The request body was rejected by the backend.',
  request_failed: 'The backend returned an unsuccessful response.',
}

/** Error panel used for both network failures and API error payloads. */
export default function ErrorState({ error, onRetry }) {
  if (!error) return null

  const hint = HINTS[error.code]

  return (
    <div className="card state-card state-card--error" role="alert">
      <div className="state-card__icon" aria-hidden="true">
        !
      </div>
      <div className="state-card__body">
        <h2 className="state-card__title">Request failed</h2>
        <p className="state-card__text">{error.message}</p>
        {hint ? <p className="state-card__meta">{hint}</p> : null}
        {error.code ? <p className="state-card__code">Error code: {error.code}</p> : null}
        {onRetry ? (
          <button type="button" className="secondary-button" onClick={onRetry}>
            Retry analysis
          </button>
        ) : null}
      </div>
    </div>
  )
}