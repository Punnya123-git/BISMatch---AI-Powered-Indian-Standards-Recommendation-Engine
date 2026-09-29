import { useCallback, useEffect, useRef, useState } from 'react'

import {
  DocumentSummary,
  EmptyState,
  ErrorState,
  Header,
  LoadingState,
  PipelineStatus,
  QueryInput,
  ResultsSection,
  TechnicalDetails,
} from '../components/index.js'
import { useAnalyzeRequirement } from '../hooks/useAnalyzeRequirement.js'
import { useBackendHealth } from '../hooks/useBackendHealth.js'
import { MIN_REQUIREMENT_LENGTH } from '../utils/constants.js'

function prefersReducedMotion() {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}

export default function HomePage() {
  const [requirement, setRequirement] = useState('')
  const [file, setFile] = useState(null)
  const [requirementError, setRequirementError] = useState('')
  const [fileError, setFileError] = useState('')

  const backend = useBackendHealth()
  const analysis = useAnalyzeRequirement()

  const inputRef = useRef(null)
  const resultsRef = useRef(null)

  const handleFileSelect = useCallback((selectedFile, validationError) => {
    setFile(selectedFile)
    setFileError(validationError || '')
  }, [])

  const handleAnalyze = useCallback(() => {
    const trimmed = requirement.trim()

    if (trimmed.length < MIN_REQUIREMENT_LENGTH) {
      setRequirementError('Please describe the requirement before analyzing.')
      inputRef.current?.focus()
      return
    }

    setRequirementError('')
    setFileError('')
    analysis.analyze({ requirement: trimmed, file })
  }, [requirement, file, analysis])

  const handleReset = useCallback(() => {
    setRequirement('')
    setFile(null)
    setRequirementError('')
    setFileError('')
    analysis.reset()
    inputRef.current?.focus()
  }, [analysis])

  const busy = analysis.isBusy
  const showResults = analysis.phase === 'success' && analysis.result

  // Bring the results into view once a run actually produced something, so the
  // user is not left looking at the input on narrow screens.
  useEffect(() => {
    if (!showResults) return

    const target = resultsRef.current
    if (!target) return

    const behavior = prefersReducedMotion() ? 'auto' : 'smooth'
    target.scrollIntoView({ behavior, block: 'start' })
  }, [showResults])

  const offline = backend.status === 'offline'

  return (
    <div className="app">
      <Header
        status={backend.status}
        error={backend.error}
        onRefresh={backend.refresh}
      />

      <main className="app__main">
        <div className="app__input">
          <QueryInput
            value={requirement}
            onChange={(next) => {
              setRequirement(next)
              if (requirementError) setRequirementError('')
            }}
            file={file}
            onFileSelect={handleFileSelect}
            fileError={fileError}
            requirementError={requirementError}
            onReset={handleReset}
            onAnalyze={handleAnalyze}
            isBusy={busy}
            inputRef={inputRef}
          />
        </div>

        <div className="app__results" ref={resultsRef} aria-live="polite">
          <PipelineStatus phase={analysis.phase} result={analysis.result} />

          {busy ? <LoadingState message={analysis.progress.message} /> : null}

          {!busy && analysis.phase === 'error' ? (
            <ErrorState error={analysis.error} onRetry={handleAnalyze} />
          ) : null}

          {!busy && showResults ? (
            <>
              <DocumentSummary upload={analysis.uploadedDocument} />
              <ResultsSection
                result={analysis.result}
                onRetry={handleAnalyze}
                onRefine={handleReset}
              />
            </>
          ) : null}

          {!busy && analysis.phase === 'idle' ? (
            <EmptyState
              variant={offline ? 'offline' : 'idle'}
              detail={offline ? backend.error : undefined}
            />
          ) : null}

          {/* Everything a developer might want, and nothing a procurement user
              needs, in one collapsed disclosure. */}
          <TechnicalDetails
            health={backend.health}
            pipeline={analysis.result?.pipeline}
            warnings={analysis.result?.warnings ?? []}
            result={analysis.result}
          />
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Powered by verified BIS standards data and AI-assisted applicability reasoning.
        </p>
        <p className="app-footer__note">
          BISMatch is an independent academic project developed for SIH. It is not operated,
          certified, or endorsed by the Bureau of Indian Standards. For the authoritative
          catalogue, search{' '}
          <a href="https://standards.bis.gov.in/" target="_blank" rel="noreferrer noopener">
            standards.bis.gov.in
          </a>
          .
        </p>
      </footer>
    </div>
  )
}