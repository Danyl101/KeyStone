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
        return probs