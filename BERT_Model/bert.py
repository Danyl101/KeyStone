import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np  

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
        return logits,probs
    
def text_encoding(text,max_text=328,step=256):
    all_encoding=[]
    for start_idx in range(0,len(text),step):
        text_snippet=text[start_idx:start_idx+max_text]
        encoding=tokenizer(text_snippet,padding=True,truncation=True,max_length=512,return_tensors="pt")
        all_encoding.append(encoding)
    return all_encoding

def logits_pass(encodings):
    all_logits=[]
    all_probs=[]
    for encoding in encodings:
        logits,probs=model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        all_logits.append(logits)
        all_probs.append(probs)
    all_logits_tensor=torch.cat(all_logits,dim=0)
    all_probs_tensor=torch.cat(all_probs,dim=0)
    return all_logits_tensor,all_probs_tensor

def average_pooling(logits,probs):
    avg_logits=torch.mean(logits,dim=0)
    avg_probs=torch.mean(probs,dim=0)
    
    values,predicted_class_logits=torch.max(avg_logits,dim=0)
    values,predicted_class_probs=torch.max(avg_probs,dim=0)
    
    class_names=["negative","neutral","positive"]
    
    predicted_label_logits = class_names[predicted_class_logits.item()]
    predicted_label_probs = class_names[predicted_class_probs.item()]
    
    return predicted_label_logits,predicted_label_probs

# Example usage:
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BERT_Classifier(num_classes=3,dropout=0.3)  # e.g. 3 classes for sentiment

    # Example input text
    text = "What's driving gold prices lower despite strong yearly gains Gold prices slipped to a two-week low as ceasefire hopes between Iran and Israel reduced safe-haven demand.Domestic rates also eased, though long-term investor interest remains strong.24, 2025, 9:43:03 AM IST (Updated)Gold prices fell to their lowest in nearly two weeks on Tuesday (June 24), after investor appetite for risk improved following US President Donald Trump's announcement of a ceasefire between Iran and Israel.Spot gold dropped 0.5% to $3,351.47 an ounce, while US gold futures declined 0.9% to $3,365.30 an ounce.The metal had earlier hit its lowest level since June 11.In India, 24-karat gold was priced at Rs1,00,680 per 10 grams.Rates for 22-karat and 18-karat gold stood at Rs92,290 and Rs75,510 per 10 grams respectively, according to data from Goodreturns.Markets are pricing in signs of de-escalation between the US and Iran, said Ilya Spivak, Head of Global Macro at Tastylive.He noted that easing geopolitical tensions are pulling investors away from traditional safe-haven assets like gold.Global equities rallied, while oil prices slipped to a one-week low.This followed Trump's announcement that both Israel and Iran had agreed to a complete ceasefire after nearly two weeks of hostilities.However, there was no confirmation from Israel, and Iran signaled that its cooperation depends on Israel halting attacks.Meanwhile, attention has shifted to the US Federal Reserve Vice Chair Michelle Bowman said rate cuts may be on the horizon due to potential job market risks.Investors now await comments from Fed Chair Jerome Powell, which could sway the outlook for interest rates.Lower rates tend to support gold by reducing the opportunity cost of holding the non-yielding asset.Despite short-term volatility, gold has surged nearly 30% so far this year, said Riya Singh, Research Analyst at Emkay Global Financial Services.She added that domestic prices rose 4% in June, even as jewellery demand softened.Central banks have also emerged as major gold buyers in recent years, reinforcing the metal's strategic value.Since 2022, institutions like the Reserve Bank of India (RBI) have steadily increased their gold reserves to diversify away from US dollar assets.This move is to help mitigate risks from currency fluctuations and potential revaluation losses, especially when the US Treasury yields rise or the dollar strengthens against other currencies,said Samit Guha, Interim CEO & Chief Financial and Technology Officer at MMTC-PAMP.Additionally, geopolitical events such as the freezing of Russian foreign reserves showed how foreign-held assets become vulnerable to sanctions and restrictions.Gold remains a borderless, safe-haven asset--central banks recognise its real value, and discerning investors should take note,he said.Domestic gold ETFs also reflect this trend.In May, Indian gold ETFs saw net inflows of Rs290 crore, with assets under management jumping 97% year-on-year to Rs62,400 crore.At the same time, imports fell 13% annually in May to $2.5 billion, reflecting cautious demand at elevated price levels.Analysts say gold prices could remain firm in the medium term, supported by lingering inflation risks, uncertain oil markets, and cautious central bank moves."
    encodings=text_encoding(text)
    
    # Forward pass
    logits,probs = logits_pass(encodings)
    
    logit_label,prob_label=average_pooling(logits,probs)


    print(logits)  # Raw prediction scores for each class
    print(probs)
    print(logit_label)
    print(prob_label)

        
        
