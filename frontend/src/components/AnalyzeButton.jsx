/** Primary call-to-action. Disabled while a request is in flight. */
export default function AnalyzeButton({
  onClick,
  disabled,
  isBusy,
  label = 'Analyze Requirement',
  busyLabel = 'Analyzing…',
}) {
  return (
    <button
      type="button"
      className={`primary-button ${isBusy ? 'primary-button--busy' : ''}`}
      onClick={onClick}
      disabled={disabled || isBusy}
      aria-busy={isBusy}
    >
      {isBusy ? <span className="spinner spinner--inline" aria-hidden="true" /> : null}
      <span>{isBusy ? busyLabel : label}</span>
    </button>
  )
}