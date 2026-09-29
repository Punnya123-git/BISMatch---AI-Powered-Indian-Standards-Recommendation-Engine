import { EXAMPLE_DOMAINS } from '../utils/constants.js'
import AnalyzeButton from './AnalyzeButton.jsx'
import DocumentUploader from './DocumentUploader.jsx'
import RequirementInput from './RequirementInput.jsx'

/**
 * The prominent query panel: requirement box, optional tender document,
 * example domain cards and the primary action.
 *
 * Existing document-upload functionality is preserved here - the file is still
 * only sent to the server when the user presses the analyse button.
 */
export default function QueryInput({
  value,
  onChange,
  file,
  onFileSelect,
  fileError,
  requirementError,
  onReset,
  onAnalyze,
  isBusy,
  inputRef,
}) {
  return (
    <section className="card query-card" aria-label="Requirement input">
      <header className="card__header">
        <div>
          <p className="card__eyebrow">Step 1</p>
          <h2 className="card__title">Describe your requirement</h2>
        </div>
        <button
          type="button"
          className="ghost-button"
          onClick={onReset}
          disabled={isBusy || (!value && !file)}
        >
          Clear
        </button>
      </header>

      <p className="query-card__lede">
        Find the Indian Standards relevant to your procurement requirement.
      </p>

      <p className="query-card__lede-support">
        Describe your procurement requirement in natural language. BISMatch identifies potential
        standards from the verified BIS knowledge base and uses AI to evaluate their
        applicability.
      </p>

      <RequirementInput
        value={value}
        onChange={onChange}
        disabled={isBusy}
        error={isBusy ? '' : requirementError}
        inputRef={inputRef}
      />

      <div className="field">
        <span className="field__label" id="example-queries-label">
          Try an example
        </span>
        <div className="domain-grid" role="group" aria-labelledby="example-queries-label">
          {EXAMPLE_DOMAINS.map((domain) => {
            const selected = value.trim() === domain.requirement
            return (
              <button
                key={domain.id}
                type="button"
                className={`domain-card ${selected ? 'domain-card--selected' : ''}`}
                disabled={isBusy}
                aria-pressed={selected}
                onClick={() => onChange(domain.requirement)}
              >
                <span className="domain-card__icon" aria-hidden="true">
                  {domain.icon}
                </span>
                <span className="domain-card__label">{domain.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      <DocumentUploader
        file={file}
        onSelect={onFileSelect}
        disabled={isBusy}
        error={isBusy ? '' : fileError}
      />

      <div className="query-card__actions">
        <AnalyzeButton
          onClick={onAnalyze}
          isBusy={isBusy}
          label="Analyze Requirement"
          busyLabel={isBusy ? 'Analyzing…' : 'Analyze Requirement'}
        />
        <p className="query-card__note">
          {isBusy
            ? 'Keep this tab open — analysis runs on the server.'
            : 'Describe your requirement in natural language — no codes or keywords required.'}
        </p>
      </div>
    </section>
  )
}