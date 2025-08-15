import { useEffect, useState } from "react"
import { PredictionGraph } from "../App.tsx"
import { lstm_return } from "api/typescript_api/lstm_api"

interface LSTM_Response{
    predictions:number[]
  }

export function LSTMGraphPage() {
  const [data, setData] = useState<LSTM_Response | null>(null)

  useEffect(() => {
    lstm_return().then(setData).catch(console.error)
  }, [])

  if (!data) return <p>Loading...</p>

  return (
    <div>
      <PredictionGraph predictions={data.predictions} />
    </div>
  )
}
