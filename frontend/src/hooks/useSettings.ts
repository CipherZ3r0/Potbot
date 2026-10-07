import { useState } from 'react'
import type { Settings } from '@/types'

const DEFAULT_SETTINGS: Settings = {
  provider: 'groq',
  model: 'llama-3.3-70b-versatile',
  retrieval_method: 'hybrid',
  use_reranking: true,
  use_query_rewriting: true,
  prompt_style: 'detailed',
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)

  const updateSettings = (patch: Partial<Settings>) => {
    setSettings((prev) => ({ ...prev, ...patch }))
  }

  return { settings, updateSettings }
}

