from torch.utils.data import Dataset,DataLoader
import os
from transformers import BertTokenizer, BertModel
from torch import nn 

train_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','train_folder')

val_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','val_folder')

test_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','test_folder')

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

class BERTDataset():
    def __init__(self,folder_path,tokenizer,max_text=328,step=256):
        self.folder_path=folder_path
        self.filepath=[os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.endswith('txt')]
        self.tokenizer=tokenizer
        self.max_text=max_text
        self.step=step
        pass
        
    def text_encoding(self,text,tokenizer,max_text=328,step=256):
        all_encoding=[]
        for start_idx in range(0,len(text),step):
            text_snippet=text[start_idx:start_idx+max_text]
            encoding=tokenizer(text_snippet,padding=True,truncation=True,max_length=512,return_tensors="pt")
            all_encoding.append(encoding)
        return all_encoding
    
    def len(self):
        return len(self.filepath)
    
    def get_item(self,idx,step=256,max_text=328):
        filepath=self.filepath[idx]
        with open(filepath ,"r")as f:
            text=f.read()
            
            all_encoding=[]
            for start_idx in range(0,len(text),step):
                text_snippet=text[start_idx:start_idx+max_text]
                encoding=tokenizer(text_snippet,padding=True,truncation=True,max_length=512,return_tensors="pt")
                all_encoding.append(encoding)
        return all_encoding
            
            

