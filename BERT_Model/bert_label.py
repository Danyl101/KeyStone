import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import numpy as np
import os
import csv
import traceback
import torch.nn.functional as F

from utils import logging

torch.manual_seed(42) #Sets seed to reduce randomness

device="cuda" if torch.cuda.is_available() else "cpu"

class_names=["negative","neutral","positive"] #Defines the sentiment classes

class BERT_Classifier(nn.Module):
    def __init__(self,num_classes=2,dropout=0.3,pretrained_model="yiyanghkust/finbert-tone"): #Defines bert encoder without classification
        super(BERT_Classifier,self).__init__()
        self.num_classes=num_classes
        self.dropout=nn.Dropout(dropout)
        self.bert=BertModel.from_pretrained(pretrained_model)
        self.classifier=nn.Linear(self.bert.config.hidden_size,num_classes)
        
    def forward(self,input_ids,attention_mask):
        output=self.bert(attention_mask=attention_mask,input_ids=input_ids)
        pooled_output=output.pooler_output
        dropped=self.dropout(pooled_output)
        logits=self.classifier(dropped)
        probs = torch.softmax(logits, dim=1)
        return probs
    
def text_encoding(text,max_text=1500,step=1000): #Function to encode text dynamically
    try:
        all_encoding=[]
        for start_idx in range(0,len(text),step):
            text_snippet=text[start_idx:start_idx+max_text] #Splits text into smaller chunks 
            encoding=tokenizer(text_snippet,padding=True,truncation=True,max_length=512,return_tensors="pt") #Encodes the split text
            all_encoding.append(encoding) #Appends all encoded values 
        logging.info("Text encoding execution successful")
    except Exception as e:
        logging.error("Execution failed text encoding")
        logging.error(traceback.format_exc())
    return all_encoding

def logits_pass(encodings): #Function to calculate sentiment of a text chunk
    try:
        agg_logits=None
        with torch.no_grad(): 
            for enc in encodings:
                input_ids=enc['input_ids'].to(device) #Moves encoded inputs into device
                attention_mask=enc['attention_mask'].to(device) #Moves encoded attention to device
                output=model(input_ids=input_ids,attention_mask=attention_mask) #Defines the above parameters into model
                logits=output.logits.squeeze(1) #Removes all dimensions of size 1 from logits 
                if agg_logits==None:
                    agg_logits=logits #Adds logits initially
                else:
                    agg_logits=agg_logits+logits #Summises logits 
                    
        probs=F.softmax(agg_logits,dim=1) #Converts logits into probability by quantifying with softmax
        pred_idx=torch.argmax(probs).item() #Returns the max sentiment value from chunk alongside its index
        return class_names[pred_idx], probs.cpu().numpy() #Returns the index and the probability
    except Exception as e:
        logging.error("Execution failed at logits pass")
        logging.error(traceback.format_exc())   

def max_pooling(probs):
    try:
        max_vals,_=torch.max(probs,dim=0)
        
        predicted_class_probs=torch.argmax(max_vals).item()
        
        predicted_label_probs = class_names[predicted_class_probs]
        logging.info("Average pooling execution successful")
    except Exception as e:
        logging.error("Execution failed at average pooling")
        logging.error(traceback.format_exc())
    
    return predicted_label_probs

def label_to_csv(filepath,label,content_dir,output_csv="Datasets/label.csv"): #Saves the textfile and its corresponding sentiment to csv
    try:
        with open(output_csv,"a",newline='')as csvfile: 
            writer=csv.writer(csvfile) 
            writer.writerow([filepath,label])  #Writes the data into csv row
        logging.info("CSV writing")
    except Exception as e:
        logging.error("Exectuion failed at writing into csv")
        logging.error(traceback.format_exc())
    
def label_extract():
    content_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','Extracted_Content') #Acquires folder containing all extracted txt files
    for filepath in os.listdir(content_dir): 
        fullpath=os.path.join(content_dir,filepath) #Finds the path of the files
        if fullpath.endswith('txt'):
            with open(fullpath,"r",encoding='utf-8')as f:
                text=f.read()  #Reads the text file
                encoded=text_encoding(text) #Breaks text into chunks and encodes them
                label,probs=logits_pass(encoded) #Acquires the sentiment and its probability
                out=label_to_csv(filepath,label,content_dir)
                print(probs) 
                print(label)         
                
# Example usage:
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone") #Loads tokenizer from pretrained finbert
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone") #Loads finbert for classification
    model.to(device)
    model.eval()
    label_extract()

        
        
