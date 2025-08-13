from lstm_utils import device,evaluate,predict,plot,set_deterministic
set_deterministic()
import optuna
import torch.optim as optim
from lstm_model import BiLSTMModel,TimeWeightedLoss,objective
import torch
from lstm_dataload import train_loader, val_loader,test_loader

#Sets a model function to run optimal model after hyperparamters are achieved from optimization
def train_model():
    model=BiLSTMModel(input_size=4, hidden_size=128, dropout=0.3, num_layers=2, batch_size=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=8.238048741701306e-05) #Defines optimizer
    criterion_train = TimeWeightedLoss()  # Example of a custom loss function
    epochs = 75
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds,_,_ = model(x) 
            loss = criterion_train(preds, y)
            
            optimizer.zero_grad() #Resets accumulated gradients from previous batch
            loss.backward() #Computes gradients of loss w.r.t. model parameters
            optimizer.step() #Updates model weights
            total_train_loss += loss.item()
    
        val_loss = evaluate(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    test_loss = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    # Optional: Get predictions and true values for plotting
    test_preds, test_actuals, mse, rmse, mae, mape = predict(model, test_loader)
    
    torch.save(model.state_dict(), f"checkpoints/nifty50_model.pt")
    
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    plot(test_preds, test_actuals)  # Plot predictions vs actual values

#Implements bayesian optimization
def optimize_hyperparameters():
    study = optuna.create_study(direction="minimize") #Initiates the optimization
    study.optimize(objective, n_trials=75,timeout=2000) #Calls optimization function 

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
if __name__=="__main__":
    train_model()
    print("Training complete.")
        