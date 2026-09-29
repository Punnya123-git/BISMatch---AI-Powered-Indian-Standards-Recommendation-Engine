import { useCallback, useEffect, useState } from 'react'

import { fetchHealth } from '../services/healthService.js'

/**
 * Poll-free backend health check with a manual refresh.
 * @returns {{health: object|null, status: 'loading'|'online'|'offline', error: string|null, refresh: Function}}
 */
export function useBackendHealth() {
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    setStatus('loading')
    setError(null)
    try {
      const payload = await fetchHealth()
      setHealth(payload)
      setStatus('online')
    } catch (cause) {
      setHealth(null)
      setError(cause.message)
      setStatus('offline')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { health, status, error, refresh }
}
