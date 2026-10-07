import { PlusCircle, X, ChevronLeft, ChevronRight } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { FileUploadTab } from '@/components/ingestion/FileUploadTab'
import { FolderIngestTab } from '@/components/ingestion/FolderIngestTab'
import { SettingsPanel } from '@/components/settings/SettingsPanel'
import type { Settings, StatusResponse } from '@/types'
import { cn } from '@/lib/utils'

interface SidebarProps {
  open: boolean
  onToggle: () => void
  settings: Settings
  onSettingsUpdate: (patch: Partial<Settings>) => void
  status: StatusResponse
  onNewChat: () => void
}

function StatItem({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="text-center">
      <div className="text-sm font-semibold text-foreground leading-tight">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5">{label}</div>
    </div>
  )
}

export function Sidebar({
  open,
  onToggle,
  settings,
  onSettingsUpdate,
  status,
  onNewChat,
}: SidebarProps) {
  const providerColor = settings.provider === 'ollama' ? '#10B981' : '#3B82F6'

  return (
    <>
      {/* Overlay on mobile */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/50 md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={cn(
          'fixed top-0 left-0 z-30 h-full w-72 bg-sidebar border-r border-border flex flex-col transition-transform duration-300 ease-in-out',
          'md:relative md:translate-x-0 md:z-auto',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-5 pb-4 border-b border-border">
          <div>
            <div className="text-base font-semibold text-foreground">potbot workspace</div>
            <div className="text-xs text-muted mt-0.5">
              Provider:{' '}
              <span className="font-semibold" style={{ color: providerColor }}>
                {settings.provider.toUpperCase()}
              </span>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="md:hidden rounded-md p-1 text-muted hover:text-foreground hover:bg-card transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-4 space-y-5">
          {/* New Chat */}
          <button
            onClick={onNewChat}
            className="flex w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-2.5 text-sm text-foreground transition-all duration-200 hover:border-accent hover:bg-card-hover hover:text-accent"
          >
            <PlusCircle className="h-4 w-4 text-accent" />
            New Chat
          </button>

          {/* Knowledge Base */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted mb-2">
              Knowledge Base
            </p>
            <Tabs defaultValue="upload">
              <TabsList>
                <TabsTrigger value="upload">Upload</TabsTrigger>
                <TabsTrigger value="folder">Folder</TabsTrigger>
              </TabsList>
              <TabsContent value="upload">
                <FileUploadTab />
              </TabsContent>
              <TabsContent value="folder">
                <FolderIngestTab />
              </TabsContent>
            </Tabs>
          </div>

          {/* Stats */}
          <div className="rounded-lg border border-border bg-card p-3">
            <div className="grid grid-cols-3 divide-x divide-border">
              <StatItem value={status.unique_file_count} label="Documents" />
              <StatItem value={status.chunk_count} label="Chunks" />
              <div className="text-center pl-2">
                <div
                  className="text-sm font-semibold leading-tight flex items-center justify-center gap-1.5"
                  style={{ color: status.online ? '#10B981' : '#EF4444' }}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: status.online ? '#10B981' : '#EF4444' }}
                  />
                  {status.status_text}
                </div>
                <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5">Status</div>
              </div>
            </div>
          </div>

          {/* Settings */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted mb-2">
              Settings
            </p>
            <SettingsPanel settings={settings} onUpdate={onSettingsUpdate} />
          </div>
        </div>

        {/* Collapse button — desktop only */}
        <button
          onClick={onToggle}
          className="hidden md:flex absolute -right-3 top-1/2 -translate-y-1/2 h-6 w-6 items-center justify-center rounded-full border border-border bg-sidebar text-muted hover:text-foreground hover:border-accent transition-all"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>
      </aside>

      {/* Collapsed sidebar toggle — desktop */}
      {!open && (
        <button
          onClick={onToggle}
          className="hidden md:flex fixed left-0 top-1/2 -translate-y-1/2 z-20 h-8 w-5 items-center justify-center rounded-r-md border border-l-0 border-border bg-sidebar text-muted hover:text-foreground hover:border-accent transition-all"
        >
          <ChevronRight className="h-3 w-3" />
        </button>
      )}
    </>
  )
}

