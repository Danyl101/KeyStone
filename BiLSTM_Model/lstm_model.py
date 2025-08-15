import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import time
import traceback
import gc

import torch.nn.functional as F
from BiLSTM_Model.lstm_dataload import train_dataset, test_dataset
from BiLSTM_Model.lstm_utils import evaluate_metrics, log_cpu_memory, device,plot
   
# 1. LSTM Model Definition
class BiLSTMModel(nn.Module):
    #Defines the structure of the LSTM model
        def __init__(self, input_size=4, hidden_size=64, num_layers=2,dropout=0.3, batch_size=32):
            super().__init__()
            try:
                self.hidden_size=hidden_size
                self.batch_size=batch_size
                self.num_layers=num_layers
                self.dropout=dropout
                self.bidirectional=True
            except Exception as e:
                logging.error(f"Error in BiLSTMModel: {e}")
                logging.debug(traceback.format_exc())
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=self.bidirectional)#Model is defined
            self.attention=Attention(hidden_size*2)
            self.dropout=nn.Dropout(dropout) #Dropout to reduce overfitting
            self.fc = nn.Linear(hidden_size*2, 1) #Produces 1 output value

        #Defines movement of data through the model
        def forward(self, x):
            out, _ = self.lstm(x)
            context,attn_weights=self.attention(out) #Feeds lstm output into attention
            out_dropped = self.dropout(context) 
            output=self.fc(out_dropped).view(-1) #Removes last dimension from output tensor
            return output,attn_weights,out  
def objective(trial):
    try:
        logging.info(f"Starting trial {trial.number}")
        start_time= time.time()
        #Defines the possible hyperparameters
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64 ,128, 256])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32 ,64])
            
        logging.info(f"Trial #{trial.number} Start Time:{start_time}  Hyperparameters: "
                    f"lr={learning_rate}, dropout={dropout}, hidden_size={hidden_size}, batch_size={batch_size}")

        #Defines loader for train and test dataset
        train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=0)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=0)
            
        log_cpu_memory("Before Trial")
            
        model = BiLSTMModel(input_size=4, hidden_size=hidden_size, dropout=dropout).to(device) #Defines model for bayesian
        train_criterion=TimeWeightedLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Defines optimizer

        epoch = 75 #No of epochs
        # Training loop (light version)
        for epoch in range(epoch):
            model.train()
            for xb, yb in train_loader: #Loads input features and target features
                xb, yb = xb.to(device), yb.to(device)               
                optimizer.zero_grad() #Resets optimizer back to zero gradient after every batch
                output,_,_ = model(xb)
                loss = train_criterion(output, yb) #Checks models output with the actual output
                loss.backward() #Propagates backwards and finds gradients of loss 
                optimizer.step() #Updates the weights

            # Evaluate
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred,_,_ = model(x)
                preds.append(pred) #Appends models predictions to the list
                truths.append(y) #Appends actual values to the list
            
        log_cpu_memory("After Trial")

        preds = torch.cat(preds, dim=0) #Converts datatype back for metrics evaluation
        truths = torch.cat(truths, dim=0)

        mse, _, _, _, = evaluate_metrics(truths, preds)
        try:
            logging.info("Directory creation underway")
            os.makedirs('Checkpoints', exist_ok=True)  # Ensure directory exists
            torch.save(model.state_dict(), f"checkpoints/best_model_trial_{trial.number}.pt")
        except Exception as e:
            logging.error(f"Directory creation failed {e}")
        #Acts as garbage collector and avoids optuna from taking existing values 
        del model
        del optimizer 
        torch.cuda.empty_cache()  # no GPU, but still clears PyTorch cache
        gc.collect()
        return mse
        
    except Exception as e:
        logging.error(f"Issue occured at trial {trial.number} {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

#Custom Loss function
class TimeWeightedLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y_pred, y_true):
            try:
                seq_len = y_true.shape[0] #Gets sequence length
                Time_weights = torch.linspace(1.0,2.0, steps=seq_len).to(y_pred.device) #Generates increasing weights
                calc_weights=torch.mean((y_pred - y_true) ** 2)*Time_weights #Calculates weights
                calc_weights_mean=calc_weights.mean() 
                return calc_weights_mean
            except Exception as e:
                logging.error(f"Error in TimeWeightedLoss: {e}")
                logging.debug(traceback.format_exc())
                return None

class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.attention=nn.Linear(hidden_dim,1) #Defines attention layer using hidden size
    
    def forward(self,lstm_output):
        energy=self.attention(lstm_output)  #Gets lstm output [B,T,1]
        weights=F.softmax(energy.squeeze(-1),dim=1) #Removes last dimension [B,T,1]-> [B,T]
        context=torch.bmm(weights.unsqueeze(1),lstm_output).squeeze(1) #Changes shape from [B,T]->[B,1,T] 
                                                                       #Multiplied with lstm out [B,1,T]->[B,1,H]
                                                                       #Squeeze remove dimension 1 [B,1,H]->[B,H]
        return context,weights
        
        
        
   
    
        

        
        
    


    



