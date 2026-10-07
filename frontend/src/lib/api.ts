/**
 * Typed API client for the potbot FastAPI backend.
 * All fetch calls go through the Vite proxy in dev (/api → localhost:8000)
 * and are served directly from nginx in production.
 */

import type {
  ChatRequest,
  ChatResponse,
  FeedbackRequest,
  FeedbackResponse,
  IngestResponse,
  StatusResponse,
} from '@/types'

const BASE = '/api'

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  status: () => request<StatusResponse>('/status'),

  supportedExtensions: () =>
    request<{ extensions: string[] }>('/supported-extensions'),

  chat: (body: ChatRequest) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  feedback: (body: FeedbackRequest) =>
    request<FeedbackResponse>('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  ingestFiles: (files: File[], recreateIndex: boolean) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    form.append('recreate_index', String(recreateIndex))
    return request<IngestResponse>('/ingest/files', { method: 'POST', body: form })
  },

  ingestFolder: (folderPath: string, recreateIndex: boolean) => {
    const form = new FormData()
    form.append('folder_path', folderPath)
    form.append('recreate_index', String(recreateIndex))
    return request<IngestResponse>('/ingest/folder', { method: 'POST', body: form })
  },
}

