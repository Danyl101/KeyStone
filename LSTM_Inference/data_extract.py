import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def extract():
    #Fetch historical data for Nifty(AAPL)
    ticker = "^NSEI"
    data = yf.download(ticker, period="70d",interval="1d")
    
     # If data has MultiIndex columns, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # keep only first level
        
    data=data.tail(60)
    print(data.head())

    # Save the data to a CSV file
    data.to_csv("LSTM_Inference/nifty_data.csv")
    
if __name__=="__main__":
    extract()