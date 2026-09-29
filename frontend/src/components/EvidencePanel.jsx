import { useState } from 'react'

import { useCopyToClipboard } from '../hooks/useCopyToClipboard.js'
import { formatScore } from '../utils/formatters.js'

function evidenceToText(item) {
  return [
    item.standard_code || item.source,
    item.page_number ? `page ${item.page_number}` : null,
    typeof item.score === 'number' ? `similarity ${formatScore(item.score)}` : null,
    '',
    item.snippet,
  ]
    .filter(Boolean)
    .join('\n')
}

/**
 * Retrieved evidence, kept deliberately separate from the AI reasoning shown on
 * each recommendation card: everything here is verbatim text returned by
 * retrieval, with its source and similarity score.
 */
export default function EvidencePanel({ items = [] }) {
  const [open, setOpen] = useState(false)
  const { copiedKey, copy } = useCopyToClipboard()

  if (!items.length) return null

  const copyAll = () => copy(items.map(evidenceToText).join('\n\n---\n\n'), 'all-evidence')

  return (
    <section className="card evidence-card" aria-label="Source passages">
      <header className="card__header">
        <div>
          <p className="card__eyebrow">Source material</p>
          <h2 className="card__title">Source passages</h2>
        </div>
        <div className="card__actions">
          <button
            type="button"
            className="ghost-button"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
          >
            {open ? 'Hide' : `Show ${items.length} passage${items.length === 1 ? '' : 's'}`}
          </button>
          {open ? (
            <button type="button" className="ghost-button" onClick={copyAll}>
              {copiedKey === 'all-evidence' ? 'Copied' : 'Copy all'}
            </button>
          ) : null}
        </div>
      </header>

      {open ? (
        <ul className="evidence-list">
          {items.map((item, index) => (
            <li key={item.chunk_id || `evidence-${index}`} className="evidence">
              <div className="evidence__head">
                <span className="evidence__source">{item.standard_code || item.source}</span>
                {item.page_number ? (
                  <span className="evidence__meta">page {item.page_number}</span>
                ) : null}
                {typeof item.score === 'number' ? (
                  <span className="evidence__meta">similarity {formatScore(item.score)}</span>
                ) : null}
                <button
                  type="button"
                  className={`copy-button ${copiedKey === `evidence-${index}` ? 'copy-button--done' : ''}`}
                  onClick={() => copy(item.snippet, `evidence-${index}`)}
                  aria-label={`Copy evidence from ${item.standard_code || item.source}`}
                  title="Copy this evidence"
                >
                  {copiedKey === `evidence-${index}` ? '✓' : '⧉'}
                </button>
              </div>
              <p className="evidence__snippet">{item.snippet}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p className="evidence-card__hint">
          {items.length} source passage{items.length === 1 ? '' : 's'} from the standards
          catalogue. These are the exact text excerpts the AI assessed.
          index. These are catalogue text, not generated text.
        </p>
      )}
    </section>
  )
}