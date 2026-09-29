import { ENDPOINTS, request } from './apiClient.js'

/**
 * Upload a PDF or text document; the backend extracts and cleans the text.
 * @param {File} file
 */
export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file, file.name)

  return request(ENDPOINTS.upload, {
    method: 'POST',
    body: formData,
    headers: { Accept: 'application/json' },
  })
}
