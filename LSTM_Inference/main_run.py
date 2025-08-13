from .data_extract import extract
from .data_process import preprocess
from BiLSTM_Model.lstm_model import evaluate_metrics,plot,run_inference
from .dataloader import true_data_loader
import torch
import matplotlib as plt
import io
import base64

model=torch.load("Checkpoints/nifty50_model.pt")

def lstm_run():
    extract()
    preprocess()
    preds,targets=run_inference(model,true_data_loader)
    mse, rmse, mae, mape=evaluate_metrics(targets,preds)
    plot(preds,targets)
    
    return {
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        },
        "predictions": preds.tolist(),
        "targets": targets.tolist(),
    }    

if __name__=="__main__":
    lstm_run()
    
    