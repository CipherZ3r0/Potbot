import { useState } from 'react'
import { Menu } from 'lucide-react'
import { Sidebar } from './Sidebar'
import { ChatArea } from '@/components/chat/ChatArea'
import { useChat } from '@/hooks/useChat'
import { useSettings } from '@/hooks/useSettings'
import { useStatus } from '@/hooks/useStatus'

export function MainLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { messages, isLoading, sendMessage, clearMessages } = useChat()
  const { settings, updateSettings } = useSettings()
  const { status } = useStatus()

  const handleSend = (query: string) => {
    sendMessage(query, settings)
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        settings={settings}
        onSettingsUpdate={updateSettings}
        status={status}
        onNewChat={clearMessages}
      />

      {/* Main content */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center gap-3 border-b border-border bg-background px-4 py-3">
          {/* Mobile menu button */}
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="md:hidden rounded-md p-1.5 text-muted hover:text-foreground hover:bg-card transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>

          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent text-xs font-bold text-white">
              pb
            </div>
            <div>
              <h1 className="text-sm font-semibold text-foreground leading-tight">potbot</h1>
              <p className="text-[11px] text-muted leading-tight">Internal Intelligence Platform</p>
            </div>
          </div>

          <div className="flex-1" />

          {/* Message count badge */}
          {messages.length > 0 && (
            <span className="text-xs text-muted">
              {Math.ceil(messages.length / 2)} exchange{Math.ceil(messages.length / 2) !== 1 ? 's' : ''}
            </span>
          )}
        </header>

        {/* Chat */}
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          settings={settings}
          onSend={handleSend}
        />
      </div>
    </div>
  )
}

