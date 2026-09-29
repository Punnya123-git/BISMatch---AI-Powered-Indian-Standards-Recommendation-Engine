import { ENDPOINTS, request } from './apiClient.js'

/** Fetch service health and which optional AI components are configured. */
export async function fetchHealth(signal) {
  return request(ENDPOINTS.health, { signal, headers: { Accept: 'application/json' } })
}
