"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Play, Database, CheckCircle, XCircle } from "lucide-react"

interface RunButtonsProps {
  onRunScrape: () => Promise<void>
  onRunExtract: () => Promise<void>
  disabled: boolean
}

export function RunButtons({ onRunScrape, onRunExtract,disabled }: RunButtonsProps) {
  const [status, setStatus] = useState<{ loading: boolean; success: boolean | null; message?: string }>({
    loading: false,
    success: null,
  })

  const runAction = async (action: () => Promise<void>, label: string) => {
    setStatus({ loading: true, success: null })
    try {
      await action()
      setStatus({ loading: false, success: true, message: `${label} completed successfully` })
    } catch (e: any) {
      setStatus({ loading: false, success: false, message: `${label} failed: ${e.message}` })
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <Button
          onClick={() => runAction(onRunScrape, "Scrape")}
          disabled={disabled || status.loading}
          size="lg"
          className="h-16 text-lg"
          variant="default"
        >
          {status.loading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Play className="h-5 w-5 mr-2" />}
          Run Scrape
        </Button>

        <Button
          onClick={() => runAction(onRunExtract, "Extract")}
          disabled={disabled || status.loading}
          size="lg"
          className="h-16 text-lg"
          variant="secondary"
        >
          {status.loading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Database className="h-5 w-5 mr-2" />}
          Run Extract
        </Button>
      </div>

      

      {status.success === true && (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">{status.message}</AlertDescription>
        </Alert>
      )}

      {status.success === false && (
        <Alert className="border-red-200 bg-red-50" variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{status.message}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}

interface LSTMButtonProps{
  onRunLSTM:() =>Promise<void>
  disabled: boolean
}

export function LSTMButtons({ onRunLSTM,disabled }: LSTMButtonProps) {
const [status, setStatus] = useState<{ loading: boolean; success: boolean | null; message?: string }>({
  loading: false,
  success: null,
})

const runAction = async (action: () => Promise<void>, label: string) => {
    setStatus({ loading: true, success: null })
    try {
      await action()
      setStatus({ loading: false, success: true, message: `${label} completed successfully` })
    } catch (e: any) {
      setStatus({ loading: false, success: false, message: `${label} failed: ${e.message}` })
    }
  }
  return(
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2"></div>
        <div className="flex justify-center">
          <Button
          onClick={() => runAction(onRunLSTM, "Run LSTM")}
          disabled={disabled || status.loading}
          size="lg"
          className="h-16 text-lg px-4 py-2 w-1/2"
          variant="default"
        >
          {status.loading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Play className="h-5 w-5 mr-2" />}
          Run Inference
        </Button>
        </div>
    </div>
)}
