export const removefilter = async (filter_remove: string[]) => { //Defines async function to acquire string from frontend
    try {
        const filter_removeRes = await fetch("http://localhost:5000/api/remove-filter", //Fetches input from web link
        { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(filter_remove),
        }); //Defines method and what to extract

        if (!filter_removeRes.ok) {
            throw new Error(`Failed to acquire filters: ${filter_removeRes.status}`); //Log message
        }
        const filter = await filter_removeRes.json(); //Converts input into json to feed into flask
        return filter;
    } catch (err: any) {
        console.error(`Failed execution at ${err.message}`, err); //Exception message
        throw err;
    }
};

export const removesite = async (sitedata: string[]) => { //Defines async function to acquire string from frontend
    try {
        const siteremoveRes = await fetch("http://localhost:5000/api/remove-site", { //Fetches input from web link
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sitedata),
        });  //Defines method and what to extract

        if (!siteremoveRes.ok) {
            throw new Error(`Failed to acquire sites: ${siteremoveRes.status}`); //Log message
        }

        const site = await siteremoveRes.json(); //Converts input into json to feed into flask
        return site;
    } catch (err: any) {
        console.error(`Failed execution at ${err.message}`, err); //Exception message
        throw err;
    }
};




