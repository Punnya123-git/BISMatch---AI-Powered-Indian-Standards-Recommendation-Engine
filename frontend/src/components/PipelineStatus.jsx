import { buildPipelineState } from '../utils/pipelineState.js'

const STATE_LABEL = {
  pending: 'Waiting',
  active: 'In progress',
  done: 'Complete',
  empty: 'No output',
  failed: 'Failed',
  skipped: 'Unavailable',
  unavailable: 'Did not complete',
}

function Marker({ state }) {
  if (state === 'active') {
    return (
      <span className="pipeline__marker" aria-hidden="true">
        <span className="spinner spinner--marker" />
      </span>
    )
  }

  const glyph =
    state === 'done' ? '✓' : state === 'unavailable' || state === 'skipped' ? '⚠' : ''

  return (
    <span className="pipeline__marker" aria-hidden="true">
      {glyph}
    </span>
  )
}

/**
 * Visualises the five pipeline stages.
 *
 * Every state is derived from the real request phase and the real response body
 * (see utils/pipelineState.js). The AI stage reads as complete only when the
 * backend reported `ranking_stage: ai_reasoning_ranking`, or when it reported a
 * completed verdict that selected nothing. Anything else is shown as "did not
 * complete" - never as a successful deterministic result.
 */
export default function PipelineStatus({ phase, result }) {
  const steps = buildPipelineState({ phase, result })

  return (
    <section className="card pipeline-card" aria-label="Analysis pipeline">
      <header className="card__header">
        <div>
          <p className="card__eyebrow">Live</p>
          <h2 className="card__title">Analysis pipeline</h2>
        </div>
      </header>

      <ol className="pipeline">
        {steps.map((item) => (
          <li key={item.id} className={`pipeline__step pipeline__step--${item.state}`}>
            <Marker state={item.state} />
            <div className="pipeline__content">
              <div className="pipeline__head">
                <span className="pipeline__label">{item.label}</span>
                <span className="pipeline__state">{STATE_LABEL[item.state]}</span>
              </div>
              {item.detail ? <p className="pipeline__detail">{item.detail}</p> : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}