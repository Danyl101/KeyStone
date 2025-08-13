import json
import os
import traceback
import time
import random

from .Selenium_newspaper import  scroll_and_extract
from .utils import is_browser_alive, setup_driver,logging
   
def extract_multiple_articles(inner_dict, max_scrolls=10):
    driver = setup_driver()  # Initialize the WebDriver
    all_articles = []
    
    for dict_content in inner_dict: #Iterates through every link in scraped list
        time.sleep(random.uniform(2, 5))
        logging.info(f"Extracting article: {dict_content['title']}")
        logging.debug(traceback.format_exc())
        try:
            if not is_browser_alive(driver): #Checks if browser is working
                driver.quit()
                driver = setup_driver()
            data = scroll_and_extract(driver, dict_content, max_scrolls=max_scrolls) #Extracts contents from articles
            all_articles.append(data)
        except:
            logging.error(f"Error extracting article: {dict_content['title']}")
            logging.debug(traceback.format_exc())
            continue
    
    driver.quit()    
    return all_articles