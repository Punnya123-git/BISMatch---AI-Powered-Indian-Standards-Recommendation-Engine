import { useState } from 'react'

/**
 * Relation categories, keyed by the *explicit* relation string in the verified
 * record. A record with no relation is grouped under "related" - never under
 * "safety" or "test", because a relationship type we cannot verify is a
 * relationship type we must not assert.
 */
const RELATION_CATEGORIES = [
  { id: 'primary', label: 'Primary', match: /primary|subject|product/i },
  { id: 'test', label: 'Test', match: /test\s*method|testing|measurement/i },
  { id: 'safety', label: 'Safety', match: /safety/i },
  { id: 'installation', label: 'Installation', match: /install|erection|commission/i },
  { id: 'terminology', label: 'Terminology', match: /terminolog|vocabulary|glossary/i },
  { id: 'related', label: 'Related', match: /related|normative|reference|referenced|informative/i },
]

function categorise(items) {
  const buckets = new Map(RELATION_CATEGORIES.map((c) => [c.id, []]))
  for (const item of items) {
    const relation = item?.relation || ''
    const match = RELATION_CATEGORIES.find((c) => c.match.test(relation))
    buckets.get(match ? match.id : 'related').push(item)
  }
  return RELATION_CATEGORIES.map((c) => ({ ...c, items: buckets.get(c.id) })).filter(
    (c) => c.items.length > 0,
  )
}

/**
 * Part 8: related and normative standards, grouped by verified relation type and
 * collapsed by category. Clicking a category reveals the standards the source
 * record explicitly names - nothing is inferred from semantic similarity.
 */
export default function RelatedStandards({ items = [] }) {
  const [open, setOpen] = useState(null)
  if (!items.length) return null

  const groups = categorise(items)

  return (
    <div className="detail detail--neutral related">
      <h5 className="detail__title">Related standards</h5>
      <p className="related__note">
        Relationships below are those explicitly recorded in the verified standard. Similarity
        alone never creates a relationship.
      </p>

      <ul className="related__groups">
        {groups.map((group) => {
          const expanded = open === group.id
          return (
            <li key={group.id} className="related__group">
              <button
                type="button"
                className={`related__toggle ${expanded ? 'related__toggle--open' : ''}`}
                aria-expanded={expanded}
                onClick={() => setOpen(expanded ? null : group.id)}
              >
                <span className="related__toggle-label">{group.label}</span>
                <span className="related__toggle-count">{group.items.length}</span>
              </button>

              {expanded ? (
                <ul className="bullet-list related__items">
                  {group.items.map((item, index) => (
                    <li key={`${item.code}-${index}`}>
                      <strong>{item.code}</strong>
                      {item.title ? <span className="muted"> — {item.title}</span> : null}
                      {item.relation ? (
                        <span className="related__relation"> ({item.relation})</span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              ) : null}
            </li>
          )
        })}
      </ul>
    </div>
  )
}