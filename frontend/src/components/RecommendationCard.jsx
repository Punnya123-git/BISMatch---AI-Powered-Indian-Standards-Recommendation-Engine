import { useState } from 'react'

import { useCopyToClipboard } from '../hooks/useCopyToClipboard.js'
import {
  APPLICABILITY_DESCRIPTIONS,
  APPLICABILITY_LABELS,
  APPLICABILITY_TONES,
} from '../utils/constants.js'
import { formatPercent, formatRank, formatScore } from '../utils/formatters.js'
import RelatedStandards from './RelatedStandards.jsx'
import StatusBadge from './StatusBadge.jsx'

function versionLabel(version) {
  if (!version) return null
  const parts = []
  if (version.year) parts.push(version.year)
  if (version.revision) parts.push(version.revision)
  if (version.status) parts.push(version.status)
  return parts.length ? parts.join(' · ') : null
}

function CopyButton({ onClick, active, label }) {
  return (
    <button
      type="button"
      className={`copy-button ${active ? 'copy-button--done' : ''}`}
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      {active ? '✓' : '⧉'}
    </button>
  )
}

function DetailBlock({ title, tone = 'neutral', children }) {
  return (
    <div className={`detail detail--${tone}`}>
      <h5 className="detail__title">{title}</h5>
      {children}
    </div>
  )
}

/**
 * One recommended standard.
 *
 * Collapsed it shows the decision-level facts (number, title, applicability,
 * confidence, AI vs retrieval rank). Expanded it reveals the full justification,
 * matched/uncovered requirements, version, certification, references and the
 * verbatim retrieved evidence - with AI prose and catalogue text kept visually
 * distinct.
 */
export default function RecommendationCard({ recommendation, rank, aiRanked, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const { copiedKey, copy } = useCopyToClipboard()

  const {
    code,
    title,
    confidence,
    category,
    applicability_type: applicabilityType,
    reasoning,
    matched_requirements: matchedRequirements = [],
    uncovered_requirements: uncoveredRequirements = [],
    deterministic_rank: deterministicRank = null,
    version,
    certification = [],
    related_standards: relatedStandards = [],
    evidence = [],
  } = recommendation

  const panelId = `rec-${code}-${rank}`.replace(/[^\w-]/g, '-')
  const versionText = versionLabel(version)
  const hasRankShift = aiRanked && typeof deterministicRank === 'number' && deterministicRank !== rank

  return (
    <article className={`rec-card ${open ? 'rec-card--open' : ''}`}>
      <div className="rec-card__head">
        <button
          type="button"
          className="rec-card__summary"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          aria-controls={panelId}
        >
          <span className="rec-card__rank" aria-hidden="true">
            {rank}
          </span>

          <span className="rec-card__headline">
            <span className="rec-card__titlerow">
              <span className="rec-card__code">{code}</span>
              <StatusBadge
                tone={APPLICABILITY_TONES[applicabilityType] || 'neutral'}
                title={APPLICABILITY_DESCRIPTIONS[applicabilityType]}
              >
                {APPLICABILITY_LABELS[applicabilityType] || applicabilityType}
              </StatusBadge>
            </span>
            {title ? <span className="rec-card__title">{title}</span> : null}

            <span className="rec-card__ranks">
              <span className="rank-chip rank-chip--ai">
                {aiRanked ? 'AI rank' : 'Rank'} {formatRank(rank)}
              </span>
              {typeof deterministicRank === 'number' ? (
                <span className="rank-chip rank-chip--retrieval">
                  Retrieved candidate {formatRank(deterministicRank)}
                </span>
              ) : null}
              {hasRankShift ? (
                <span className="rank-chip rank-chip--shift">
                  moved {deterministicRank > rank ? 'up' : 'down'}{' '}
                  {Math.abs(deterministicRank - rank)}
                </span>
              ) : null}
            </span>
          </span>

          <span className="rec-card__score">
            <span className="rec-card__score-value">{formatPercent(confidence)}</span>
            <span className="rec-card__score-label">confidence</span>
          </span>

          <span
            className={`rec-card__chevron ${open ? 'rec-card__chevron--open' : ''}`}
            aria-hidden="true"
          >
            ▾
          </span>
        </button>

        <CopyButton
          label={`Copy standard number ${code}`}
          active={copiedKey === `code-${code}`}
          onClick={() => copy(code, `code-${code}`)}
        />
      </div>

      <div className="rec-card__meter" aria-hidden="true">
        <span className="rec-card__meter-fill" style={{ width: formatPercent(confidence) }} />
      </div>

      {open ? (
        <div className="rec-card__body" id={panelId}>
          <DetailBlock title="Why it matches" tone="ai">
            <div className="detail__row">
              <p className="prose">{reasoning}</p>
              <CopyButton
                label={`Copy reasoning for ${code}`}
                active={copiedKey === `reasoning-${code}`}
                onClick={() => copy(reasoning, `reasoning-${code}`)}
              />
            </div>
            <p className="detail__origin">
              {aiRanked
                ? 'Generated by the configured LLM from the retrieved evidence below.'
                : 'Template text built from catalogue data — no LLM verdict was used for this run.'}
            </p>
          </DetailBlock>

          {matchedRequirements.length ? (
            <DetailBlock title="Matched requirements">
              <ul className="tag-list">
                {matchedRequirements.map((item) => (
                  <li key={item} className="tag-pill tag-pill--match">
                    {item}
                  </li>
                ))}
              </ul>
            </DetailBlock>
          ) : null}

          {uncoveredRequirements.length ? (
            <DetailBlock title="Uncovered requirements" tone="warning">
              <ul className="tag-list">
                {uncoveredRequirements.map((item) => (
                  <li key={item} className="tag-pill tag-pill--gap">
                    {item}
                  </li>
                ))}
              </ul>
            </DetailBlock>
          ) : null}

          {versionText || certification.length ? (
            <div className="detail-grid">
              {versionText ? (
                <DetailBlock title="Version / revision">
                  <p className="prose">{versionText}</p>
                  {/* Part 9: a verified newer edition is announced, never silently
                      swapped in - the user must see which edition was evaluated. */}
                  {version?.latest_revision ? (
                    <p className="detail__flag">
                      <strong>Newer BIS revision available:</strong>{' '}
                      {version.latest_revision}
                    </p>
                  ) : null}
                  {version?.amendments?.length ? (
                    <ul className="bullet-list">
                      <li className="bullet-list__heading">Amendments</li>
                      {version.amendments.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  ) : null}
                </DetailBlock>
              ) : null}

              {/* Part 10: certification is shown only when the verified record
                  states it. The existence of a standard never implies that
                  certification is mandatory, so that is never inferred. */}
              <DetailBlock title="Certification">
                {certification.length ? (
                  <ul className="bullet-list">
                    {certification.map((item, index) => (
                      <li key={`${item.scheme || 'scheme'}-${index}`}>
                        {[item.scheme, item.marking, item.notes].filter(Boolean).join(' · ')}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="detail__missing">
                    Not stated in the verified record. Whether certification is mandatory or
                    voluntary is not stated in our source data — confirm against the official BIS
                    catalogue before relying on it.
                  </p>
                )}
              </DetailBlock>
            </div>
          ) : null}

          {relatedStandards.length ? (
            <RelatedStandards items={relatedStandards} />
          ) : null}

          {evidence.length ? (
            <DetailBlock title="Source passage" tone="evidence">
              <ul className="evidence-list evidence-list--compact">
                {evidence.map((item, index) => (
                  <li key={item.chunk_id || `evidence-${index}`} className="evidence">
                    <div className="evidence__head">
                      <span className="evidence__source">
                        {item.standard_code || item.source}
                      </span>
                      {typeof item.score === 'number' ? (
                        <span className="evidence__meta">
                          similarity {formatScore(item.score)}
                        </span>
                      ) : null}
                      <CopyButton
                        label={`Copy evidence for ${code}`}
                        active={copiedKey === `evidence-${code}-${index}`}
                        onClick={() => copy(item.snippet, `evidence-${code}-${index}`)}
                      />
                    </div>
                    <p className="evidence__snippet">{item.snippet}</p>
                  </li>
                ))}
              </ul>
              <p className="detail__origin">
                Verbatim catalogue text returned by retrieval — not generated.
              </p>
            </DetailBlock>
          ) : null}

          {category ? <p className="detail__origin">Catalogue category: {category}</p> : null}
        </div>
      ) : null}
    </article>
  )
}