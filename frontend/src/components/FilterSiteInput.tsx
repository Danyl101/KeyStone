"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Plus, Minus, CheckCircle, XCircle } from "lucide-react"

interface FilterSiteInputProps {
  label: string
  onAdd: (input:string) => Promise<void>
  onRemove: (input:string) => Promise<void>
  disabled: boolean
}

export function FilterSiteInput({ label, onAdd, onRemove, disabled }: FilterSiteInputProps) {
  const [input, setInput] = useState("")
  const [status, setStatus] = useState<{ loading: boolean; success: boolean | null; message?: string }>({
    loading: false,
    success: null,
  })

  const handleAdd = async () => {
    setStatus({ loading: true, success: null })
    try {
      await onAdd(input)
      setStatus({ loading: false, success: true, message: `${label} "${input}" added successfully` })
      setInput("")
    } catch (e: any) {
      setStatus({ loading: false, success: false, message: `Add failed: ${e.message}` })
    }
  }

  const handleRemove = async () => {
    setStatus({ loading: true, success: null })
    try {
      await onRemove(input)
      setStatus({ loading: false, success: true, message: `${label} "${input}" removed successfully` })
      setInput("")
    } catch (e: any) {
      setStatus({ loading: false, success: false, message: `Remove failed: ${e.message}` })
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor={`${label.toLowerCase()}-input`}>{label} URL or Pattern</Label>
        <Input
          id={`${label.toLowerCase()}-input`}
          type="text"
          placeholder={`Enter ${label.toLowerCase()} URL or pattern...`}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={disabled || status.loading}
          className="w-full"
        />
      </div>

      <div className="flex gap-2">
        <Button
          onClick={handleAdd}
          disabled={disabled || status.loading || !input.trim()}
          className="flex-1"
          variant="default"
        >
          {status.loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Plus className="h-4 w-4 mr-2" />}
          Add {label}
        </Button>
        <Button
          onClick={handleRemove}
          disabled={disabled || status.loading || !input.trim()}
          className="flex-1"
          variant="destructive"
        >
          {status.loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Minus className="h-4 w-4 mr-2" />}
          Remove {label}
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
