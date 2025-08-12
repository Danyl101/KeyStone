                                    POST CONSTRUCTION MODEL LOG


# Tags:[LSTM,TCN,SCRAPER,EXTRACT,BILSTM,API,REACT,BERT]
___________________________

# Default Model [TCN]

So the initial issue was validation loss was too high ,training loss(0.8) validation loss(3.1) ,and final MSE value was around (8) , this meant that the model was not generalising properly or that validation and test datasets had wildy varying values compared to training dataset , when manually reviewed this seemed to be the case , as the validation and test values had around 2 point difference to training values , training value(-1.2) , validation value (1.8) test value (2.4), this was deduced to be due to the fact that nifty 50 is a rapidly growing indice and training values were acquired from a slow bullish-bearish market and covid collapse , while validation was taken from a strong bullish market and test was taken from very strong bullish market

Training Loss(0.8)
Validation Loss(3.1)
MSE(8.3)

___________________________

# Iteration 1 [TCN]

Initially the dataset was scaled in group and then split in the model program , this would mean that even though it increases accuracy a bit  (as validation and test values are reduced to match training to a degree) , it would not hold in real life as they all should be treated as independent entities to simulate real life market movements , 
the splitting caused even a sharper drop with validation loss and mse , with training loss being (0.8) validation loss being (6.2) and mse being (25) 

Training Loss(0.9)
Validation Loss(6.3)
MSE(25)

___________________________

# Iteration 2  [TCN]

Standard scaler would take in outliers at face value and not smooth them , which made the already large variances present between the training and validation and test datasets even larger , this issue was addressed by replacing standardscaler with robustscaler which evened out the outliers thus reducing variances in certain values but not solving overall issues 

No large differences in any values

__________________________

# Iteration 3 [Log Transformation] [TCN]

Since it was groups of large values that were causing issues , log transformation was applied which reduced validation and test dataset values a by a good margin , most of train dataset were left intact due to most of it being negative thus skipping log transformation or being very small ,thus reducing loss and bringing MSE values closer to necessary requirement , but a concurrent issue facing all iterations till now has been the steady loss values , instead of dropping as epochs increases which indicates model learning , it is staying constant which means that model is not learning and is just blindly predicting 

Training Loss (0.5)
Validation Loss(1.2)
MSE(6.1)

_____________________________

# Switched Model [LSTM]

Since the TCN model was constantly returning large loss and MSE values no matter how standardized and preprocessed the dataset was ,so after going through various research papers , i decided to test out a basic lstm system , since it was much more potent at understanding long sequences of data, after moving to just a basic lstm model, it could predict basic movement of the market but not the magnitude , as it really only understands temporal features it could capture movements but not the magnitude of these moves ,after it predicted the movements to a certain degree i moved onto the scraper 

Training Loss (0.5)
Validation Loss (0.8)
MSE(0.11)

This model worked , and returned a good enough output temporarily leaving it ,to build the rest of the system , will come back add technical indicators and polish it once the entire system is complete 

____________________________

# Iteration 1 [BILSTM]

Decided to upgrade the lstm to a bilstm , since the bilstm would be able to capture much more temporal dependencies being of capable understanding both previous and future patterns from a certain point , also since training and validation,test having such a different values was a consistent issue , decided to add a custom loss function to the training dataset while keeeping original loss function for validationa and test

Training Loss (0.2)
Validation Loss(0.5)
MSE(0.05)

____________________________

# Iteration 2 [BILSTM]

So there was an apparent issue with close being fed into validation and test dataset as well , which lead to a small data leakage , which skewed the results of the final model , after correcting this the results of final model were and added other metrics along mse for various different cases ,

Training Loss (0.2)
Validation Loss(0.5)
MSE: 0.0727, RMSE: 0.2696, MAE: 0.2419, R²: -0.5629, MAPE: 9.99%

__________________________

# Iteration 3 [BILSTM]

Since the model now had somewhat good results and the overall architecture was good , i decided to begin fine tuning the hyperparameters , for this i implemented bayesian optimization ,initially wanted to do gaussian functions by myself but since that requires some time and acquisition function could be difficult ,i decided to go for optuna , but there was an issue where optuna constantly became stuck ,so added in a bunch of try catch blocks everywhere and added return(inf) to the optuna block which solved it

__________________________

# Iteration 4 [BILSTM]

So i hit a black box issue in a way , basically optuna was getting stuck on random processes and sometimes it ran no trials , sometimes 2 and sometimes 3 , it was incredibly inconsistent, i added cpu logging to see if ram usage was an issue , logging also didnt work much because there was no explcit issue in code but more so inner workings failure , decided to set up a specific seed (42) to rule out any randomness and thus hopefully decrease random interal bugs but that didnt work well , also removed the manual runs before the optuna runs to save time

__________________________

# Iteration 5 [BILSTM]

So the issue was that embarassingly i didnt save the models after each trials , and also ran the manual train on every iteration which was unnecessary , so after fixing those , the optuna began working well and gave solid results
it initially ran at t-1 which gave unsurprisingly near pinpoint results so increased gap t-20 and then finally t-45
which is where the model stands at right now , the model was run on google colab due to its gpu accomodation

Parameters
input_size=4, hidden_size=128, dropout=0.3, num_layers=2, batch_size=16 lr=5.401798979253006e-05 
epochs=75

Epoch [75/75] Train Loss: 4.9970 | Val Loss: 0.4735
Test Loss: 0.6706
MSE: 0.0550, RMSE: 0.2345, MAE: 0.2030, R²: 0.0449, MAPE: 8.31%

    Model is saved in Checkpoints folder 

___________________________

# Iteration 6 [BILSTM]

So decided to try and add an attention mechanism to the model ,it initially produced diminshing results due to custom loss function being moreso a batch weighting that acted as timeweights due to unshuffling , but i decided to truly put timeweightedloss and ran it with the attention mechanism and their complimentary effects boosted the overall metrics, also ran bayesian on this and found a new output
    
        energy=self.attention(lstm_output)  #B,T,1
        weights=F.softmax(energy.squeeze(-1),dim=1)   #B,T
        context=torch.bmm(weights.unsqueeze(1),lstm_output).squeeze(1) #B,H

Parameters 
input_size=4, hidden_size=256, dropout=0.3, num_layers=2, batch_size=32 lr=8.238048741701306e-05
epochs=75

Epoch [75/75] Train Loss: 5.2792 | Val Loss: 0.5012
Test Loss: 0.3741
MSE: 0.0299, RMSE: 0.1730, MAE: 0.1493, MAPE: 6.08%

__________________________
__________________________


# Default Model [SCRAPER]

The default model used a beautiful soup model, that initially looked for keywords in queries ,as in searched the links itself for the certain keywords , which didnt work, then switched to google rss to get certain articles or links but since google rss only showed a limited amount of articles we switched from that as well

___________________________

# Iteration 1 [SCRAPER]

We swtiched the scraper to instead of checking google rss or we accessed certain economic sites link and searched for the keywords within these site links , which narrowed down the articles it has to parse through and get more results 

__________________________

# Iteration 2 [SCRAPER]

Beautiful soup only returned the static pages urls and links ,so the links it could acquire were limited since most sites nowadays used script for these links , to access these dynamic pages , we switched from beautiful soup to selenium which can scroll or move through these sites and access all the links present 

__________________________

# Iteration 3 [SCRAPER]

Fully utilizing Selenium now , instead of checking individual tags for certain keywords or links to articles , since BERT is a model that benefits from having "junk" fed to it , we instead removed all select loops and just grabbed every href link on all the pages , this returned around 1200 articles ,from economic times and livemint, moneycontrol had an antibot software "akami" that prevented us from scraping it , and since overcoming that obstacle would question the legality and ethics , i opted not to 

__________________________

# Iteration 4 [SCRAPER]

Since BERT requires a large volume of data , i decided to add more sites to scrape for the model , initially i decided to do this manually ,just adding links to url and then building the for loop myself , i realised it would take alot of time , so instead decided to automate the process ,uttilizing two arrays , one for the website links and one for the saved articles, i also added a bert scroller to this t access more links that might not have been previously accessible just through loading javascript using selenium

__________________________

# Iteration 5 [SCRAPER]

Added a blacklist array which indicates the sites to not scrape, this was done so that bert does not scrape or access junk sites , also added alot of try exception blocks so that program dosent crash when a single site dosent load

This iteration worked , added a few exception blocks here and there, added functions to check browser health , and make the system more redundant

__________________________

# Iteration 6 [SCRAPER]

So modularized the code and split it into smaller chunks

link_extract contains the main functions which extracts links from the sites 

robot contains the function that parses through robot.txt of sites

utils contains the supporting function that are not integral but are general purpose 

Scrape_run contains the main run program

___________________________

# Iteration 7 [SCRAPER]

So changed the data loading sequence , since if frontend have to manipulate the filters and sites then they have to be present outside of a python program,so created a new json file that contains the filters and sites , it was two initially but that could cause issues in frontend so made it one , added new functions to scraper to load these json files as needed 
___________________________

___________________________

# Default Model [EXTRACTOR]

The default model followed a similar architecture to that of the scraper since they both had a similar function , but this one used newpapery3k to parse through the articles whose links and titles were collected and stored in a json file , the loader is built to extract all the content in main articles and store it in a seperate folder as txt files with their respective title names 

__________________________

# Iteration 1 [EXTRACTOR]

Since newspapery3k class article was constantly reloaded on every scroll but the contents were not stored to a file or array,we lost all the data except the content that was present on the html doc on the last iteration or the last scroll , to fix this we implemented a string that is constantly concatenated on every loop of the scroll ,as in article writes its content into this string before new scroll is executed 

__________________________

# Iteration 2 [EXTRACTOR]

Sinces articles load the same url constantly even after scrolling the page that is loaded is the initial page ,so scrolling becomes irrelevant ,to prevent this we needed to store the links in an array and after every scroll append the new link to this list and have the article url be called afterwards , so that the url can be skipped and article skips all url that came beforehand ,which are the ones present in array 

Scroll-->Click-->Article loaded-->Content parsed-->content stored-->Scroll again
                                |
                                |
                                |
Scroll-->Click-->Url Stored-->Url check runs-->Article loads-->Content parsed-->Content stored-->Scroll again

___________________________

# Iteartion 3 [EXTRACTOR]

Build an advanced get function which creats a link with the site and receives logs from there , thus allowing more advanced debugging, added debugging into every single layer of the program and found the core issues which was a pathing issue , where new spaces caused error in the windows file naming system , updated the sanitizer of filenames  

___________________________

# Iteration 4 [EXTRACTOR]

Ran into issues where some sites where parsed properly and some werent ,this was caused by the issue that most sites dont use proper tags and instead just use " " and strong tags inside html for paragraphs , which breaks most libraries used for parsing , created a seperate function for dealing with such raw html tags by utilizing xpath 

___________________________

# Iteration 5 [EXTRACTOR]

Abandoned the xpath function and instead decided to use a new library playwright , which is capable of rendering and parsing even the most js heavy and complicated and with a few tweaks , it was capable of parsing through the most broken html sites , but since it was somewhat slow it is used alongside selenium-newspaper,as in certain links are fed into playwright and rest are fed into newspaper
___________________________

# Iteration 6 [EXTRACTOR]

So playwright began working for me , but certain articles became stuck on parsing for extremely long times ,so added a timeout function for playwright , also split everything into modules and its own seperate programs, there was a tiny issue with the loader where if the link of the site wasnt explicitly named then it would automatically go to selenium-newspaper which could cause issues , so instead of checking site name at loader , every link is sent to selenium , and if it fails then its sent to playwright

                                            Sucess
            LOADER---->SELENIUM-NEWSPAPER------------>FILE SAVED
                            |
                            | Fail
                            |
                        PLAYWRIGHT----->FILE SAVED
___________________________

# Iteration 7 [EXTRACTOR]

So i modularized the code , and split it into smaller chunks 

Utils file containing general functions that are not integral to code and act as supporting code

Selenium_newspaper contains the newspaper-selenium function that is used to load the content that can be extracted from non js heavy sites 

Playwright_extract contains the code to extract content from JS-heavy sites  

Content_Extract contains the main function that goes through site after site

Extract_Run Contains the main run program

___________________________
___________________________

# Default Model [API]

Created new json files where one stores goodlist and other stores websites , this was done so that api can directly interact with these lists which were previously inside .py files which could not be accessed 

Created save and fetch function for both of these json files so that they can be updated and viewed

Initially the api program was its own seperate folder but due to access issues for react it was moved inside the frontend

___________________________________

# Iteration 1 [API]

Created ts api that interacts with the frontend , since frontend cannot directly with python files had to built it using typescript , it contains fetch functions for viewing the lists and websites that were added , and save function for changing the lists and websites from the frontend itself , these were done through async functions to increase reliability

___________________________________

# Iteration 2 [API]

Advanced the apis especially typescript to receive inputs from the browser, changed the funcionalities a bit , now instead of adding and viewing only , we instead can now add filters and sites view them as well as remove any ones , these were all done in async , ran into a few glitches here and there , so added logging to the entire flask api 
___________________________________

# Iteration 3 [API]

Modularized the code ,split the flask api into two addition api and removal api and the same was also done to typescript apis , build a new run program for flask to avoid manually running the individual flask files ,but ran into some issues here due to the pathing issues 

___________________________________

# Iteration 4 [API]

The pathing issues were caused by parent directory being executing files directory , solved this by setting a python path and then running program ,also added two new api calls one to call scraper and one to call extractor
each executing their respective python files 

___________________________________

# Default Model [REACT]

So changed the components section as in list editor and site editor to be aligned with the app program ,set up interfaces on both of them to properly define the parameters in functions , in App created an interface to properly define the data import coming in from api , and changed the function of all keyword (site,list,setsites,setlist) from  being a simple usestate to being a string

___________________________________


# Iteration 1 [REACT]

Decidied to stop development of gui until the api functionalities were all working , decided to develop a barebones structure with each api call having a button and sharing the same textfield , gui will be further developed once all the api calls are properly working

___________________________________

# Iteration 2 [REACT]

So since all the apis calls were working properly decided to move forward with react ui and build a minimal and clean ui that is enough for demo representation , further additions will be made once the full system is complete and mostly just to add polish


__________________________________
__________________________________

# Default Model [BERT]

So since my ui was nearing completeion i decided to begin on the bert model , the initial issue was preprocessing which was done through 3 standard functions and 2 custom functions , the 3 standard were regex->cleantext->spacy
and the 2 custom was deleting certain files that contain keywords in quick sucession and another to remove some trailing spaces

Then built a simple finbert model which had dynamic text tokenizing due to text files being too long and then was fed into the pretrained finbert model which was pre loaded with 3 classes(positive,neutral,negative) and the output from that was average pooled to clearly indicate what the sentiment of  text file is

_________________________________

# Iteration 1 [BERT]

Decided to fine tune the finbert myself , so created a data splitter that the splits all the txt files present  into 3 (train,test,val) , then created a dataloader that takes the text from these files and reused the tokenizer and encodings used previously to encode these properly ,increased the text size to a higher amount to reduce the padding present inside each vector and thus reduce padding noise  ,currently working on a labeler that labels all the files that were collected into 3 classes to act as the supervised data for finbert to train on ,but ran into slight issues with the labeler being biased 







