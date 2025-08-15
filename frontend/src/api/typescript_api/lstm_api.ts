export const lstm_run = async() =>{
    try{
        const response=await fetch(
            "http://localhost:5000/api/run-lstm",
        {  
            method:"POST",
            headers:{"Content-Type":"Application/json"},
        })
        if(!response.ok)
        {
            throw new Error(`Response not acquired from frontend ${response.status}`)
        }
        const action=await response.json()
        return action
    }
    catch(err:any)
    {
        console.log(`Execution failed during lstm api ${err.message}`,err)
        throw new err
    }
}

export interface LSTM_Response{
    metrics:{
        mse:number
        rmse:number
        mae:number
        mape:number
    }
    predictions:number[]
    targets:number[]
}

export const lstm_return = async(): Promise<{
    predictions: number[];
}> =>{
    const response=await fetch(
        "http://localhost:5000/api/lstm-run"
    )
    if(!response.ok)
    {
        throw new Error(`Data never acquired from lstm ${response.status}`)
    }
    return await response.json()
}


