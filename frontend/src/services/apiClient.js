/**
 * Single place where the browser learns how to reach the backend.
 *
 * Default behaviour is "same origin": in development Vite proxies /api to the
 * FastAPI server (see vite.config.js), in production the built assets are
 * served next to the API. Setting VITE_API_BASE_URL overrides this.
 */
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

export const API_PREFIX = `${API_BASE_URL}/api`

export const ENDPOINTS = {
  health: `${API_PREFIX}/health`,
  upload: `${API_PREFIX}/documents/upload`,
  analyze: `${API_PREFIX}/recommendations/analyze`,
}

/** Error type that preserves the backend error contract (code + message). */
export class ApiError extends Error {
  constructor(message, { code = 'request_failed', status = 0 } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

async function parseBody(response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return { message: text }
  }
}

/**
 * Thin fetch wrapper: JSON in, JSON out, backend errors surfaced as ApiError.
 * @param {string} url
 * @param {RequestInit} [options]
 */
export async function request(url, options = {}) {
  let response
  try {
    response = await fetch(url, options)
  } catch (cause) {
    throw new ApiError(
      'Could not reach the backend. Is the FastAPI server running on port 8000?',
      { code: 'network_error' },
    )
  }

  const body = await parseBody(response)

  if (!response.ok) {
    throw new ApiError(body?.message || `Request failed (${response.status})`, {
      code: body?.code || 'request_failed',
      status: response.status,
    })
  }

  return body
}
