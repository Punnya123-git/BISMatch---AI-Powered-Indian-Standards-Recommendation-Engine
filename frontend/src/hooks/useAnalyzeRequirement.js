import { useCallback, useState } from 'react'

import { uploadDocument } from '../services/documentService.js'
import { analyzeRequirement } from '../services/recommendationService.js'

const IDLE = { phase: 'idle', message: '' }

/**
 * Runs the upload + analyse flow and exposes it as explicit phases so the UI can
 * show exactly which step is running (or which step failed).
 *
 * Phases: idle | uploading | analysing | success | error
 */
export function useAnalyzeRequirement() {
  const [phase, setPhase] = useState(IDLE.phase)
  const [progress, setProgress] = useState(IDLE)
  const [result, setResult] = useState(null)
  const [uploadedDocument, setUploadedDocument] = useState(null)
  const [error, setError] = useState(null)

  const isBusy = phase === 'uploading' || phase === 'analysing'

  const reset = useCallback(() => {
    setPhase(IDLE.phase)
    setProgress(IDLE)
    setResult(null)
    setUploadedDocument(null)
    setError(null)
  }, [])

  /**
   * @param {{requirement: string, file?: File|null, topK?: number|null}} input
   */
  const analyze = useCallback(async ({ requirement, file, topK }) => {
    setError(null)
    setResult(null)
    setUploadedDocument(null)

    try {
      let documentId = null

      if (file) {
        setPhase('uploading')
        setProgress({ phase: 'uploading', message: `Reading ${file.name}…` })
        const upload = await uploadDocument(file)
        documentId = upload.metadata.document_id
        setUploadedDocument(upload)
      }

      setPhase('analysing')
      setProgress({
        phase: 'analysing',
        message: 'Finding relevant standards and running AI applicability analysis…',
      })
      const response = await analyzeRequirement({ requirement, documentId, topK })
      setResult(response)
      setPhase('success')
      setProgress(IDLE)
      return response
    } catch (cause) {
      setError(cause)
      setPhase('error')
      setProgress(IDLE)
      return null
    }
  }, [])

  return {
    phase,
    progress,
    isBusy,
    error,
    result,
    uploadedDocument,
    analyze,
    reset,
  }
}
