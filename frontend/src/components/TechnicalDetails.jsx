import { formatCount } from '../utils/formatters.js'

function Row({ label, value, tone }) {
  return (
    <div className="fact-row">
      <span className="fact-row__label">{label}</span>
      <span className={`fact-row__value ${tone ? `fact-row__value--${tone}` : ''}`}>{value}</span>
    </div>
  )
}

/**
 * Developer-facing diagnostics, collapsed and out of the way.
 *
 * Everything a procurement user does not need - providers, vector store, chunk
 * counts, dataset version, retrieval stage identifiers and the backend's own
 * machine warnings - lives in here and nowhere else. The normal view is a
 * product, not a dashboard.
 */
export default function TechnicalDetails({ health, pipeline, warnings = [], result }) {
  const dataset = health?.dataset ?? null
  const hasContent =
    health || pipeline || warnings.length || result?.ai_reasoning || result?.pipeline
  if (!hasContent) return null

  return (
    <details className="card tech">
      <summary className="tech__summary">
        <span className="tech__title">Technical details</span>
        <span className="tech__hint">Retrieval backend, index size and API diagnostics</span>
      </summary>

      <div className="tech__body">
        {pipeline ? (
          <section className="tech__section">
            <h3 className="tech__heading">Retrieval engine</h3>
            <dl className="fact-list">
              <div className="fact-row">
                <span className="fact-row__label">LLM provider</span>
                <span className="fact-row__value">{pipeline.llm_provider || '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Embedding provider</span>
                <span className="fact-row__value">{pipeline.embedding_provider || '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Vector store</span>
                <span className="fact-row__value">{pipeline.vector_store || '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Indexed chunks</span>
                <span className="fact-row__value">{pipeline.indexed_chunks ?? '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Standards loaded</span>
                <span className="fact-row__value">{pipeline.standards_available ?? '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Ranking stage</span>
                <span className="fact-row__value">{pipeline.ranking_stage || '—'}</span>
              </div>
            </dl>
            {pipeline.reasons?.length ? (
              <ul className="reason-list">
                {pipeline.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            ) : null}
          </section>
        ) : null}

        {result?.ai_reasoning ? (
          <section className="tech__section">
            <h3 className="tech__heading">AI reasoning stage</h3>
            <dl className="fact-list">
              <div className="fact-row">
                <span className="fact-row__label">Status</span>
                <span className="fact-row__value">{result.ai_reasoning.status}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Model</span>
                <span className="fact-row__value">{result.ai_reasoning.model || '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Candidates supplied</span>
                <span className="fact-row__value">{result.ai_reasoning.candidates_supplied}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Candidates selected</span>
                <span className="fact-row__value">{result.ai_reasoning.candidates_selected}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Truncated</span>
                <span className="fact-row__value">{String(result.ai_reasoning.truncated)}</span>
              </div>
            </dl>
            {result.ai_reasoning.invented_dropped?.length ? (
              <p className="tech__note">
                Dropped as outside the verified candidate set:{' '}
                {result.ai_reasoning.invented_dropped.join(', ')}
              </p>
            ) : null}
          </section>
        ) : null}

        {health ? (
          <section className="tech__section">
            <h3 className="tech__heading">API health</h3>
            <dl className="fact-list">
              <div className="fact-row">
                <span className="fact-row__label">Version</span>
                <span className="fact-row__value">
                  {health.version} · {health.environment}
                </span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Dataset version</span>
                <span className="fact-row__value">{dataset?.dataset_version || '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Product category</span>
                <span className="fact-row__value">{dataset?.product_category || '—'}</span>
              </div>
              <div className="fact-row">
                <span className="fact-row__label">Organization</span>
                <span className="fact-row__value">{dataset?.organization || '—'}</span>
              </div>
            </dl>
          </section>
        ) : null}

        {warnings.length ? (
          <section className="tech__section">
            <h3 className="tech__heading">Backend diagnostics</h3>
            <ul className="reason-list">
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    </details>
  )
}