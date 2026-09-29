/**
 * Backend warnings and limitations, shown verbatim and never filtered or
 * reworded: fallbacks, provider failures, truncated responses, discarded
 * designations and unmatched attributes all come straight from the API.
 */
export default function WarningsPanel({ warnings = [], title = 'Notes & limitations' }) {
  if (!warnings.length) return null

  return (
    <section className="card warnings-card" aria-label="Warnings and limitations" role="note">
      <header className="card__header">
        <div>
          <p className="card__eyebrow">Quality notes</p>
          <h2 className="card__title">{title}</h2>
        </div>
        <span className="count-chip count-chip--warning">{warnings.length}</span>
      </header>

      <ul className="warning-list">
        {warnings.map((warning, index) => (
          <li key={`${index}-${warning.slice(0, 24)}`} className="warning">
            <span className="warning__marker" aria-hidden="true">
              !
            </span>
            <p>{warning}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}