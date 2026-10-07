import { useEffect, useState } from 'react'
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from '@/components/ui/toast'

type ToastVariant = 'default' | 'success' | 'destructive'

interface ToastItem {
  id: string
  title?: string
  description?: string
  variant?: ToastVariant
}

let addToast: (toast: Omit<ToastItem, 'id'>) => void = () => {}

export function toast(opts: Omit<ToastItem, 'id'>) {
  addToast(opts)
}

export function Toaster() {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  useEffect(() => {
    addToast = (opts) => {
      const id = Math.random().toString(36).slice(2)
      setToasts((prev) => [...prev, { id, ...opts }])
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
      }, 4000)
    }
  }, [])

  return (
    <ToastProvider>
      {toasts.map((t) => (
        <Toast key={t.id} variant={t.variant}>
          {t.title && <ToastTitle>{t.title}</ToastTitle>}
          {t.description && <ToastDescription>{t.description}</ToastDescription>}
          <ToastClose />
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  )
}

