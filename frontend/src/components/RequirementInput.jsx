import { MAX_REQUIREMENT_LENGTH, MIN_REQUIREMENT_LENGTH } from '../utils/constants.js'

const PLACEHOLDER =
  'Describe the product, material or work being procured in plain language — e.g. "15 kW IE3 three-phase squirrel cage induction motor, 415 V, 50 Hz, IP55". Include ratings, voltages, capacities, materials, service conditions or test expectations and the engine will map them to the Indian Standards that apply.'

/**
 * Primary free-text requirement box.
 *
 * Controlled component: the page owns the value, this only renders + validates.
 */
export default function RequirementInput({ value, onChange, disabled, error, inputRef }) {
  const tooShort = value.trim().length > 0 && value.trim().length < MIN_REQUIREMENT_LENGTH
  const message = error || (tooShort ? 'Please enter at least a few words.' : '')

  return (
    <div className="field">
      <label className="field__label" htmlFor="requirement">
        Procurement / product requirement
        <span className="field__hint">
          Natural language works best — technical attributes are extracted and matched against the
          catalogue.
        </span>
      </label>

      <textarea
        id="requirement"
        ref={inputRef}
        className={`field__input field__input--large ${message ? 'field__input--invalid' : ''}`}
        rows={7}
        placeholder={PLACEHOLDER}
        value={value}
        maxLength={MAX_REQUIREMENT_LENGTH}
        disabled={disabled}
        aria-invalid={message ? 'true' : 'false'}
        aria-describedby={message ? 'requirement-error' : undefined}
        onChange={(event) => onChange(event.target.value)}
      />

      <div className="field__footer">
        <span
          id="requirement-error"
          className={`field__error ${message ? 'field__error--visible' : ''}`}
          role="alert"
        >
          {message}
        </span>
        <span className="field__counter">
          {value.length} / {MAX_REQUIREMENT_LENGTH}
        </span>
      </div>
    </div>
  )
}