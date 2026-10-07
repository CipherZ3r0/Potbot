import { useState, useRef } from 'react'
import { Upload, RotateCcw, CheckCircle2, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import { toast } from '@/components/ui/toaster'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export function FileUploadTab() {
  const [files, setFiles] = useState<File[]>([])
  const [rebuild, setRebuild] = useState(false)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name))
      const newFiles = Array.from(incoming).filter((f) => !names.has(f.name))
      return [...prev, ...newFiles]
    })
  }

  const handleIngest = async () => {
    if (!files.length) {
      toast({ variant: 'destructive', title: 'No files', description: 'Please select files first.' })
      return
    }
    setLoading(true)
    try {
      const result = await api.ingestFiles(files, rebuild)
      if (result.indexed_count > 0) {
        toast({
          variant: 'success',
          title: 'Ingestion complete',
          description: `Indexed ${result.indexed_count} chunks from ${result.doc_count} document(s).`,
        })
        setFiles([])
      } else {
        toast({ variant: 'default', title: 'Nothing indexed', description: `Status: ${result.status}` })
      }
    } catch (err) {
      toast({
        variant: 'destructive',
        title: 'Ingestion failed',
        description: err instanceof Error ? err.message : 'Unknown error',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          'flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed p-6 text-center transition-all duration-200',
          dragging
            ? 'border-accent bg-accent/5'
            : 'border-border bg-card hover:border-accent hover:bg-card-hover',
        )}
      >
        <Upload className={cn('h-6 w-6', dragging ? 'text-accent' : 'text-muted')} />
        <p className="text-xs text-muted">
          {files.length > 0
            ? `${files.length} file(s) selected`
            : 'Drop files or click to browse'}
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => addFiles(e.target.files)}
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <ul className="space-y-1 max-h-32 overflow-y-auto scrollbar-thin">
          {files.map((f) => (
            <li
              key={f.name}
              className="flex items-center justify-between gap-2 rounded-md bg-card px-2 py-1.5 text-xs text-muted"
            >
              <span className="truncate flex-1">{f.name}</span>
              <button
                onClick={() => setFiles((prev) => prev.filter((x) => x.name !== f.name))}
                className="text-muted hover:text-red-400 transition-colors"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Rebuild toggle */}
      <label className="flex items-center gap-2 text-xs text-muted cursor-pointer select-none">
        <input
          type="checkbox"
          checked={rebuild}
          onChange={(e) => setRebuild(e.target.checked)}
          className="accent-accent"
        />
        <RotateCcw className="h-3 w-3" />
        Rebuild index
      </label>

      <Button
        onClick={handleIngest}
        disabled={loading || !files.length}
        className="w-full"
        variant={files.length ? 'accent' : 'default'}
      >
        {loading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Processing…
          </>
        ) : (
          <>
            <CheckCircle2 className="h-3.5 w-3.5" />
            Ingest {files.length > 0 ? `${files.length} file(s)` : 'Files'}
          </>
        )}
      </Button>
    </div>
  )
}

