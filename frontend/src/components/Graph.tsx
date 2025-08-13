import { useEffect, useState } from "react"
import { PredictionGraph } from "../App.tsx"
import { lstm_return } from "api/typescript_api/lstm_api.ts"
import { LSTMResponse } from "api/typescript_api/lstm_api.ts"

export function LSTMGraphPage() {
  const [data, setData] = useState<LSTMResponse | null>(null)

  useEffect(() => {
    lstm_return().then(setData).catch(console.error)
  }, [])

  if (!data) return <p>Loading...</p>

  return (
    <div>
      <h2>Metrics</h2>
      <ul>
        <li>MSE: {data.metrics.mse}</li>
        <li>RMSE: {data.metrics.rmse}</li>
        <li>MAE: {data.metrics.mae}</li>
        <li>MAPE: {data.metrics.mape}</li>
      </ul>

      <PredictionGraph predictions={data.predictions} targets={data.targets} />
    </div>
  )
}
