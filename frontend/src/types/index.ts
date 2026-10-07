// Shared TypeScript interfaces mirroring the FastAPI Pydantic schemas

export interface SourceItem {
  file_name: string
  page_number: number | null
  text: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[]
  conversationId?: number
  responseTimeMs?: number
  totalTokens?: number
  model?: string
  isError?: boolean
  isLoading?: boolean
}

export interface ChatRequest {
  query: string
  retrieval_method: string
  use_reranking: boolean
  use_query_rewriting: boolean
  prompt_style: string
  llm_provider: string
  llm_model: string
}

export interface ChatResponse {
  answer: string
  sources: SourceItem[]
  conversation_id: number | null
  response_time_ms: number
  total_tokens: number
  model: string
  retrieval_method: string
  prompt_style: string
}

export interface FeedbackRequest {
  conversation_id: number
  sentiment: 'positive' | 'negative'
  comment?: string
}

export interface FeedbackResponse {
  feedback_id: number
  message: string
}

export interface StatusResponse {
  online: boolean
  status_text: string
  unique_file_count: number
  chunk_count: number
}

export interface IngestResponse {
  status: string
  indexed_count: number
  doc_count: number
  chunk_count: number
}

export interface Settings {
  provider: 'groq' | 'ollama'
  model: string
  retrieval_method: 'hybrid' | 'vector' | 'text'
  use_reranking: boolean
  use_query_rewriting: boolean
  prompt_style: 'detailed' | 'concise' | 'structured'
}

