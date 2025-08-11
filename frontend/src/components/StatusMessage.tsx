"use client"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, CheckCircle, XCircle } from "lucide-react"

interface StatusMessageProps {
  loading: boolean
  success: boolean | null
  message?: string
}

export function StatusMessage({ loading, success, message }: StatusMessageProps) {
  if (loading) {
    return (
      <Alert className="border-blue-200 bg-blue-50">
        <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
        <AlertDescription className="text-blue-800">Processing...</AlertDescription>
      </Alert>
    )
  }

  if (success === true) {
    return (
      <Alert className="border-green-200 bg-green-50">
        <CheckCircle className="h-4 w-4 text-green-600" />
        <AlertDescription className="text-green-800">{message}</AlertDescription>
      </Alert>
    )
  }

  if (success === false) {
    return (
      <Alert className="border-red-200 bg-red-50" variant="destructive">
        <XCircle className="h-4 w-4" />
        <AlertDescription>{message}</AlertDescription>
      </Alert>
    )
  }

  return null
}
