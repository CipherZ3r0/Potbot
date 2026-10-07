import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import type { StatusResponse } from '@/types'

const POLL_INTERVAL_MS = 30_000

const INITIAL_STATUS: StatusResponse = {
  online: false,
  status_text: 'Connecting…',
  unique_file_count: 0,
  chunk_count: 0,
}

export function useStatus() {
  const [status, setStatus] = useState<StatusResponse>(INITIAL_STATUS)
  const [loading, setLoading] = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.status()
      setStatus(data)
    } catch {
      setStatus((prev) => ({ ...prev, online: false, status_text: 'Offline' }))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [fetchStatus])

  return { status, loading, refetch: fetchStatus }
}

