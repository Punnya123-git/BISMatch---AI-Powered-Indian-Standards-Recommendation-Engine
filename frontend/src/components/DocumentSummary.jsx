import { formatBytes, formatCount } from '../utils/formatters.js'

/**
 * Summary of the tender document the backend parsed, shown after a successful
 * upload + analysis so the user can confirm the right file was used.
 */
export default function DocumentSummary({ upload }) {
  if (!upload) return null

  const { metadata, document } = upload
  const warnings = document?.warnings ?? []

  return (
    <section className="card doc-card" aria-label="Processed document">
      <header className="card__header">
        <div>
          <p className="card__eyebrow">Processed document</p>
          <h2 className="card__title">{metadata?.filename}</h2>
        </div>
      </header>

      <dl className="stat-grid stat-grid--compact">
        <div className="stat-grid__item">
          <dt>Document id</dt>
          <dd className="stat-grid__value--text">{metadata?.document_id}</dd>
        </div>
        <div className="stat-grid__item">
          <dt>Pages</dt>
          <dd>{document?.page_count ?? '—'}</dd>
        </div>
        <div className="stat-grid__item">
          <dt>Words</dt>
          <dd>{formatCount(document?.word_count, 'word')}</dd>
        </div>
        <div className="stat-grid__item">
          <dt>Size</dt>
          <dd>{formatBytes(metadata?.size_bytes)}</dd>
        </div>
      </dl>

      {warnings.length ? (
        <ul className="warning-list">
          {warnings.map((warning, index) => (
            <li key={`${index}-${warning.slice(0, 20)}`} className="warning">
              <span className="warning__marker" aria-hidden="true">
                !
              </span>
              <p>{warning}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  )
}