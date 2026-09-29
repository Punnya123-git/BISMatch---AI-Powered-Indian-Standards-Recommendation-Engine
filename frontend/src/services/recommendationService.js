import { ENDPOINTS, request } from './apiClient.js'

/**
 * Ask the backend to analyse a requirement (with an optional uploaded document).
 * The response always carries an explicit `status` so the UI never has to guess
 * whether recommendations are real or the pipeline is still unconfigured.
 *
 * @param {{requirement: string, documentId?: string|null, topK?: number|null}} params
 */
export async function analyzeRequirement({ requirement, documentId, topK }) {
  return request(ENDPOINTS.analyze, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({
      requirement,
      document_id: documentId || null,
      top_k: topK ?? null,
    }),
  })
}
