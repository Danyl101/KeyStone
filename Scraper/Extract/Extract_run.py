import json
import os
import logging

from .content_extract import extract_multiple_articles

with open('Datasets/scraped_article.json', 'r') as f: #Loads json file 
    article = json.load(f) 
    json_dict=article["articles"] #articles is dictionary inside the json file ,this loads the actual articles into json_dict

    os.makedirs('BERT_Content', exist_ok=True)  # Ensure directory exists

def extract_run():
    extracted_articles = extract_multiple_articles(json_dict, max_scrolls=10) #Calls the complete extraction process
    logging.info(f"Total Extracted articles:{len(extracted_articles)}")
    
if __name__=="__main__":
    extract_run()
    
    
    
    
    
        
    
    

