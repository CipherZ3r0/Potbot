import { useState } from 'react'
import type { Settings } from '@/types'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'

interface SettingsPanelProps {
  settings: Settings
  onUpdate: (patch: Partial<Settings>) => void
}

const GROQ_MODELS = [
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'mixtral-8x7b-32768',
  'gemma2-9b-it',
]

const OLLAMA_MODELS = ['llama3', 'mistral', 'gemma2', 'phi3']

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-muted mb-2 mt-4 first:mt-0">
      {children}
    </p>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1.5">
      <span className="text-xs text-muted">{label}</span>
      <div className="flex-shrink-0">{children}</div>
    </div>
  )
}

export function SettingsPanel({ settings, onUpdate }: SettingsPanelProps) {
  const [customModel, setCustomModel] = useState('')
  const [useCustom, setUseCustom] = useState(false)

  const models = settings.provider === 'groq' ? GROQ_MODELS : OLLAMA_MODELS
  const isKnownModel = models.includes(settings.model)

  const handleModelChange = (val: string) => {
    if (val === '__custom__') {
      setUseCustom(true)
    } else {
      setUseCustom(false)
      onUpdate({ model: val })
    }
  }

  return (
    <div className="space-y-0.5">
      {/* Provider & Model */}
      <SectionLabel>Provider &amp; Model</SectionLabel>

      <Row label="Provider">
        <Select
          value={settings.provider}
          onValueChange={(v) =>
            onUpdate({
              provider: v as 'groq' | 'ollama',
              model: v === 'groq' ? GROQ_MODELS[0] : OLLAMA_MODELS[0],
            })
          }
        >
          <SelectTrigger className="h-8 text-xs w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="groq">Groq (Cloud)</SelectItem>
            <SelectItem value="ollama">Ollama (Local)</SelectItem>
          </SelectContent>
        </Select>
      </Row>

      <Row label="Model">
        <Select
          value={useCustom ? '__custom__' : (isKnownModel ? settings.model : '__custom__')}
          onValueChange={handleModelChange}
        >
          <SelectTrigger className="h-8 text-xs w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {models.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
            <SelectItem value="__custom__">Custom…</SelectItem>
          </SelectContent>
        </Select>
      </Row>

      {(useCustom || !isKnownModel) && (
        <Input
          value={customModel || (!isKnownModel ? settings.model : '')}
          onChange={(e) => {
            setCustomModel(e.target.value)
            onUpdate({ model: e.target.value })
          }}
          placeholder="Enter model name"
          className="h-8 text-xs mt-1"
        />
      )}

      {/* Search Strategy */}
      <SectionLabel>Search Strategy</SectionLabel>

      <Row label="Method">
        <Select
          value={settings.retrieval_method}
          onValueChange={(v) => onUpdate({ retrieval_method: v as Settings['retrieval_method'] })}
        >
          <SelectTrigger className="h-8 text-xs w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="hybrid">Hybrid (BM25 + kNN)</SelectItem>
            <SelectItem value="vector">Vector only</SelectItem>
            <SelectItem value="text">BM25 only</SelectItem>
          </SelectContent>
        </Select>
      </Row>

      <Row label="Rerank results">
        <Switch
          checked={settings.use_reranking}
          onCheckedChange={(v) => onUpdate({ use_reranking: v })}
        />
      </Row>

      <Row label="Rewrite query">
        <Switch
          checked={settings.use_query_rewriting}
          onCheckedChange={(v) => onUpdate({ use_query_rewriting: v })}
        />
      </Row>

      {/* Output */}
      <SectionLabel>Output Style</SectionLabel>

      <Row label="Style">
        <Select
          value={settings.prompt_style}
          onValueChange={(v) => onUpdate({ prompt_style: v as Settings['prompt_style'] })}
        >
          <SelectTrigger className="h-8 text-xs w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="detailed">Detailed</SelectItem>
            <SelectItem value="concise">Concise</SelectItem>
            <SelectItem value="structured">Structured</SelectItem>
          </SelectContent>
        </Select>
      </Row>
    </div>
  )
}

