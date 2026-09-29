import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Copy-to-clipboard with a short "Copied" confirmation.
 *
 * `navigator.clipboard` is unavailable on insecure origins, so a hidden textarea
 * + `document.execCommand` path keeps the buttons working in every context.
 *
 * @param {number} [resetAfter] ms before the confirmation clears
 */
export function useCopyToClipboard(resetAfter = 1800) {
  const [copiedKey, setCopiedKey] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => () => clearTimeout(timerRef.current), [])

  const copy = useCallback(
    async (text, key = null) => {
      if (typeof text !== 'string' || text.length === 0) return false

      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(text)
        } else {
          const helper = document.createElement('textarea')
          helper.value = text
          helper.setAttribute('readonly', '')
          helper.style.position = 'fixed'
          helper.style.opacity = '0'
          document.body.appendChild(helper)
          helper.select()
          const ok = document.execCommand('copy')
          document.body.removeChild(helper)
          if (!ok) return false
        }

        setCopiedKey(key)
        clearTimeout(timerRef.current)
        timerRef.current = setTimeout(() => setCopiedKey(null), resetAfter)
        return true
      } catch {
        return false
      }
    },
    [resetAfter],
  )

  return { copiedKey, copy }
}