/** Small presentation helpers (pure functions, easy to unit test). */

export function formatBytes(bytes) {
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function formatCount(value, singular, plural = `${singular}s`) {
  if (typeof value !== 'number') return '—'
  return `${value.toLocaleString('en-IN')} ${value === 1 ? singular : plural}`
}

export function formatConfidence(confidence) {
  if (typeof confidence !== 'number') return '—'
  return `${Math.round(confidence * 100)}%`
}

export function formatScore(score) {
  if (typeof score !== 'number') return '—'
  return score.toFixed(3)
}

export function formatDateTime(value) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

export function truncate(text, maxLength = 220) {
  if (typeof text !== 'string') return ''
  const trimmed = text.trim()
  if (trimmed.length <= maxLength) return trimmed
  return `${trimmed.slice(0, maxLength).trimEnd()}…`
}

/** 0.9 -> "90%". Used for confidence and relevance scores. */
export function formatPercent(value, digits = 0) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** 10 -> "#10". Ranks are 1-based; anything else renders as an em dash. */
export function formatRank(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return `#${value}`
}

/**
 * Join a short list inline, collapsing the tail so dense arrays (matched
 * requirements, related standards) never break a card layout.
 */
export function joinInline(items = [], max = 3) {
  if (!Array.isArray(items) || items.length === 0) return ''
  if (items.length <= max) return items.join(', ')
  return `${items.slice(0, max).join(', ')} +${items.length - max} more`
}

/** "1 standard" / "4 standards" without the "1 standards" bug. */
export function pluralize(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`
}
