export const addfilter = async (filterdata: string[]) => {  //Defines async function to acquire string from frontend
    try {
        const filterRes = await fetch("http://localhost:5000/api/add-filter", //Fetches input from web link
        { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(filterdata),
        }); //Defines method and what to extract

        if (!filterRes.ok) {
            throw new Error(`Failed to acquire filters: ${filterRes.status}`); //Log message
        }

        const filter = await filterRes.json(); //Converts input into json to feed into flask
        return filter;
    } catch (err: any) {
        console.error(`Failed execution at ${err.message}`, err); //Exception msg
        throw err;
    }
};

export const addsite = async (sitedata: string[]) => { //Defines async function to acquire string from frontend
    try {
        const siteRes = await fetch("http://localhost:5000/api/add-site", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sitedata),
        }); //Defines method and what to extract

        if (!siteRes.ok) {
            throw new Error(`Failed to acquire sites: ${siteRes.status}`); //Log message
        }

        const site = await siteRes.json(); //Converts input into json to feed into flask
        return site;
    } catch (err: any) {
        console.error(`Failed execution at ${err.message}`, err); //Exception msg
        throw err;
    }
};