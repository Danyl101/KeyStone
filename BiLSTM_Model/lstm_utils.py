import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging
import traceback
import psutil
import os

#Logging function for optuna trials
logging.basicConfig(
        filename="Optuna.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )

#Sets a seed to reduce randomness
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#Logs the cpu memory during trials 
def log_cpu_memory(tag=""):
        process = psutil.Process(os.getpid())#Acquires process
        mem_mb = process.memory_info().rss / 1024 ** 2  # in MB
        logging.info(f"[{tag}] CPU RAM Usage: {mem_mb:.2f} MB")
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
#Function to evaluate metrics
def evaluate_metrics(y_true, y_pred):
    try:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()#Detaches array from cpu and converts into numpy
            
        if isinstance(y_true,torch.Tensor):
            y_true =y_true.detach.cpu().numpy()
        
        mse = mean_squared_error(y_true, y_pred) #Calculate Metrics
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100
    except Exception as e:
        logging.error(f"Error in evaluate_metrics: {e}")
        logging.debug(traceback.format_exc())
        return None, None, None, None

    return mse, rmse, mae, mape #Returns various metrics
    
#Validation function (Validation data)
def evaluate(model, loader):
    log_cpu_memory("Before Trial")
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:#Loads arguments from sequences
            x, y = x.to(device), y.to(device) 
            preds,_,_= model(x) #Passes the sequences through the model
            loss = criterion(preds, y) #Calculates the loss
            total_loss += loss.item()
            
    log_cpu_memory("After Trial")        
    return total_loss

#Prediction function(Test Data)
def predict(model, loader):
    model.eval()
    preds, targets = [], [] #Defines empty lists to store predictions and targets
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred,_,_ = model(x)
            preds.append(pred) #Appends models predictions to the list
            targets.append(y) #Appends actual values to the list
                    
        preds = torch.cat(preds, dim=0) #Converts datatype back for metrics evaluation
        targets = torch.cat(targets, dim=0)

    mse, rmse, mae, mape = evaluate_metrics(preds, targets)
    return preds,targets,mse,rmse,mae,mape
    
def plot(test_prediction,test_actuals):
#Plotting the predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.plot(test_prediction, label='Predicted')
    plt.plot(test_actuals, label='Actual')
    plt.title("LSTM Model Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Scaled Close Price")        
    plt.legend()
    plt.savefig(r'D:\Prediction_Model\Documentation\BiLSTM Graph_2.png')
    

        
    
