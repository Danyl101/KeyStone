from collections import Counter
import os
import csv
import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers import MarianMTModel,MarianTokenizer

device="cuda" if torch.cuda.is_available() else "cpu"

en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"

tokenizer_en_fr=MarianTokenizer.from_pretrained(en_fr_model_name)
model_en_fr=MarianMTModel.from_pretrained(en_fr_model_name).to(device)

tokenizer_fr_en=MarianTokenizer.from_pretrained(fr_en_model_name)
model_fr_en=MarianMTModel.from_pretrained(fr_en_model_name).to(device)

labels=[]
with open("Datasets/label.csv","r")as csvfile:
    reader=csv.reader(csvfile)
    header=next(reader)
    for row in reader:
        labels.append(row[1])
    
counts=Counter(labels)
print(counts)
negative_num=counts["negative"]
neutral_num=counts["neutral"]
positive_num=counts["positive"]


class_counts=torch.tensor([negative_num,neutral_num,positive_num],dtype=torch.float)
class_weights=class_counts/1.0
class_weights=class_weights/class_weights.sum()
criterion=CrossEntropyLoss(weight=class_weights)

def text_acquire():
    content_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','Extracted_Content')
    all_text=[]
    all_filename=[]
    for filename in os.listdir(content_dir):
        filepath=os.path.join(content_dir,filename)
        with open(filepath,"r",encoding="utf-8")as f:
            text=f.read()
            all_text.append(text)
            all_filename.append(filename)
    return all_text,all_filename
    
def text_encoding(texts,max_text=1500,step=1000):
    all_text=[]
    for text in texts:
        all_snippets=[]
        for start_idx in range(0,len(text),step):
            text_snippet=text[start_idx:start_idx+max_text]
            all_snippets.append(text_snippet)
        all_text.append(all_snippets)
    return all_text

def translate(texts):
    backtranslated_chunks = []
    for text in texts:
        single_page_text=[]
        for chunk in text:
            # English -> French
            enc = tokenizer_en_fr(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            fr_out = model_en_fr.generate(**enc)
            fr_text = tokenizer_en_fr.batch_decode(fr_out, skip_special_tokens=True)
            torch.cuda.empty_cache()

            # French -> English
            enc_back = tokenizer_fr_en(fr_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc_back = {k: v.to(device) for k, v in enc_back.items()}
            en_out = model_fr_en.generate(**enc_back)
            en_text = tokenizer_fr_en.batch_decode(en_out, skip_special_tokens=True)
            torch.cuda.empty_cache()

            single_page_text.append(en_text)
        backtranslated_chunks.append(single_page_text)
    return backtranslated_chunks

def save_file(content,filename):
    for text,name in zip(content,filename):
        full_text="".join(text)
        name_only, ext = os.path.splitext(name)  # splits "file.txt" -> ("file", ".txt")
        changed_filename = name_only + "_french.txt"
        filepath=os.path.join('BERT_Content',changed_filename)
        with open(filepath, "w",encoding="utf-8")as f:
            f.write(full_text)

if __name__=="__main__":
        texts,filename=text_acquire()
        encoding=text_encoding(texts)
        translated=translate(encoding)
        save_file(translated,filename)
        
    
    
    


    

    
        