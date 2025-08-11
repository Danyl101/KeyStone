export const scrape_run= async() =>{ // Defines async function to get response from frontend
    try{    
        const response=await fetch(
            "http://localhost:5000/api/run-scrape", //Fetches input from web link
            {
                method:"POST",
                headers:{"Content-Type":"application/json"}
            });
            if(!response.ok)
            {
                throw new Error(`Response not received ${response.status}`) //Log message
            }
            const data=await response.json() //Converts response into json to transfer to flask
            return data
        }
        catch(err:any)
        {
            console.log(`Server Response error ${err.message}`,err) //Exception msg
            throw err
        }
}

export const extract_run= async() =>{ // Defines async function to get response from frontend
    try{
    const response=await fetch(
        "http://localhost:5000/api/run-extract",//Fetches input from web link
        {
            method:"POST",
            headers:{"Content-Type":"application/json"},
        }
    );
    if(!response.ok)
    {
        throw new Error(`Response not received ${response.status}`) //Log message
    }
    const data=await response.json() //Converts response into json to transfer to flask
    return data
    }
    catch(err:any)
    {
        console.log(`Server Response failed ${err.message}`,err) //Exception msg
        throw err
    }
}
