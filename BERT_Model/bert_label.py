import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np
import os
import csv
import traceback

from utils import logging

torch.manual_seed(42)

class BERT_Classifier(nn.Module):
    def __init__(self,num_classes=2,dropout=0.3,pretrained_model="yiyanghkust/finbert-tone"):
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
    
def text_encoding(text,max_text=1500,step=1000):
    try:
        all_encoding=[]
        for start_idx in range(0,len(text),step):
            text_snippet=text[start_idx:start_idx+max_text]
            encoding=tokenizer(text_snippet,padding=True,truncation=True,max_length=512,return_tensors="pt")
            all_encoding.append(encoding)
        logging.info("Text encoding execution successful")
    except Exception as e:
        logging.error("Execution failed text encoding")
        logging.error(traceback.format_exc())
    return all_encoding

def logits_pass(encodings):
    try:
        all_probs=[]
        for encoding in encodings:
            probs=model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
            all_probs.append(probs)
        all_probs_tensor=torch.cat(all_probs,dim=0)
        logging.info("Logits pass execution successful")
    except Exception as e:
        logging.error("Execution failed at logits pass")
        logging.error(traceback.format_exc())
    return all_probs_tensor

def max_pooling(probs):
    try:
        max_vals,_=torch.max(probs,dim=0)
        
        predicted_class_probs=torch.argmax(max_vals).item()
        
        class_names=["negative","neutral","positive"]
        
        predicted_label_probs = class_names[predicted_class_probs]
        logging.info("Average pooling execution successful")
    except Exception as e:
        logging.error("Execution failed at average pooling")
        logging.error(traceback.format_exc())
    
    return predicted_label_probs

def label_to_csv(filepath,label,content_dir,output_csv="Datasets/label.csv"):
    try:
        with open(output_csv,"a",newline='')as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([filepath,label])
        logging.info("CSV writing")
    except Exception as e:
        logging.error("Exectuion failed at writing into csv")
        logging.error(traceback.format_exc())
    
def label_extract():
    content_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','Extracted_Content') 
    for filepath in os.listdir(content_dir):
        fullpath=os.path.join(content_dir,filepath)
        if fullpath.endswith('txt'):
            with open(fullpath,"r",encoding='utf-8')as f:
                text=f.read()
                encoded=text_encoding(text)
                probs=logits_pass(encoded)
                label=max_pooling(probs)
                out=label_to_csv(filepath,label,content_dir)
                print(probs)
                print(label)            
                
# Example usage:
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BERT_Classifier(num_classes=3,dropout=0.3)  # e.g. 3 classes for sentiment
    model.eval()
    label_extract()

        
        
