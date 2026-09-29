import { useRef, useState } from 'react'

import { ACCEPTED_DOCUMENT_TYPES, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB } from '../utils/constants.js'
import { formatBytes } from '../utils/formatters.js'

/**
 * Drag-and-drop / click file picker for the tender document.
 * The file is only uploaded when the user presses Analyse, so choosing the
 * wrong document never leaves junk on the server.
 */
export default function DocumentUploader({ file, onSelect, disabled, error }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const accept = (candidate) => {
    if (!candidate) return
    if (candidate.size > MAX_UPLOAD_SIZE_BYTES) {
      onSelect(null, `"${candidate.name}" is larger than the ${MAX_UPLOAD_SIZE_MB} MB limit.`)
      return
    }
    onSelect(candidate, null)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) return
    accept(event.dataTransfer.files?.[0])
  }

  return (
    <div className="field">
      <span className="field__label">
        Tender / specification document
        <span className="field__hint"> Optional. PDF or plain text, up to {MAX_UPLOAD_SIZE_MB} MB.</span>
      </span>

      <div
        className={`dropzone ${isDragging ? 'dropzone--active' : ''} ${disabled ? 'dropzone--disabled' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => {
          if (!disabled) inputRef.current?.click()
        }}
        onKeyDown={(event) => {
          if (disabled) return
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            inputRef.current?.click()
          }
        }}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
      >
        <input
          ref={inputRef}
          type="file"
          className="dropzone__input"
          accept={ACCEPTED_DOCUMENT_TYPES}
          disabled={disabled}
          onChange={(event) => accept(event.target.files?.[0])}
        />

        {file ? (
          <div className="dropzone__file">
            <span className="dropzone__file-name">{file.name}</span>
            <span className="dropzone__file-meta">
              {formatBytes(file.size)} · click to replace
            </span>
          </div>
        ) : (
          <div className="dropzone__prompt">
            <strong>Drag a PDF here</strong>
            <span>or click to browse your files</span>
          </div>
        )}
      </div>

      <div className="field__footer">
        <span className="field__error" role="alert">
          {error || ''}
        </span>
        {file ? (
          <button
            type="button"
            className="link-button"
            disabled={disabled}
            onClick={() => onSelect(null)}
          >
            Remove file
          </button>
        ) : null}
      </div>
    </div>
  )
}
