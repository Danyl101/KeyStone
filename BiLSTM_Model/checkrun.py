from BiLSTM_Model.lstm_utils import device,evaluate,predict,plot,set_deterministic
set_deterministic()
import optuna
import torch.optim as optim
from BiLSTM_Model.lstm_model import BiLSTMModel,TimeWeightedLoss,objective,run_inference
import torch
from BiLSTM_Model.lstm_dataload import train_loader, val_loader,test_loader
    
    
def run_inference(model, dataloader, device=None, return_targets=True):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=BiLSTMModel(input_size=4, hidden_size=256, dropout=0.3, num_layers=2, batch_size=32).to(device)
    state_dict=torch.load(r"Checkpoints\nifty50_model.pt")
    model.load_state_dict(state_dict)
    model.to(device) 

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds_batch,_,_ = model(x)
            preds.append(preds_batch)
            if return_targets:
                targets.append(y)
            
    preds=torch.cat(preds,dim=0).cpu().numpy()
    targets=torch.cat(targets,dim=0).cpu().numpy()
    plot(targets,preds)
    if return_targets:
        return preds, targets
    else:
        return preds
    
if __name__=="__main__":
    model=1
    preds,targets=run_inference(model,train_loader)
    
    

