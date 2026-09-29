import { useEffect, useState } from 'react'

/** Phase-aware loading indicator with a live elapsed timer. */
export default function LoadingState({ message = 'Working…' }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setElapsed((value) => value + 1), 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="card loading-card" role="status" aria-live="polite">
      <div className="loading-card__bar" aria-hidden="true">
        <span />
      </div>
      <div className="loading-card__body">
        <span className="spinner" aria-hidden="true" />
        <div>
          <p className="loading-card__title">{message}</p>
          <p className="loading-card__meta">
            Running on the server · {elapsed}s elapsed. This usually takes 10–30 seconds while the
            AI reviews the standards found in the catalogue.
          </p>
        </div>
      </div>
    </div>
  )
}