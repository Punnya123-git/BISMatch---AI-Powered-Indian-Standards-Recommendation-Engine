/**
 * Small pill used for statuses, applicability types and ranking stages.
 * `tone` maps to the shared palette in index.css.
 */
export default function StatusBadge({ tone = 'neutral', children, title, className = '' }) {
  const classes = ['badge', `badge--${tone}`, className].filter(Boolean).join(' ')

  return (
    <span className={classes} title={title}>
      <span className="badge__dot" aria-hidden="true" />
      {children}
    </span>
  )
}