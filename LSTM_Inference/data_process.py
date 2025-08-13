import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from BiLSTM_Preprocess.lstm_dataprocess import scaler

def preprocess():
    #Accessing the necessary features 
    featurescale=['Close','High','Low','Open','Volume']

    data=pd.read_csv('LSTM_Inference/nifty_data.csv')
    print("Data after transformation:")
    print(data.head())
    #Defining the scaler
    lookback=60

    fit_data=data[featurescale]

    #Scaling the data
    scaleddata= scaler.fit_transform(fit_data)
    final_scaled_data = pd.DataFrame(scaleddata, columns=featurescale)

    # Inserting the date column if it exists in the original data
    if 'Date' in data.columns:
        final_scaled_data.insert(0, 'Date', data['Date'].reset_index(drop=True))

    # Displaying the first few rows of the scaled data
    print(final_scaled_data.head())

    # Saving the scaled data to a CSV file
    final_scaled_data.to_csv('LSTM_Inference/nifty_scaled.csv', index=False)

if __name__=="__main__":
    preprocess()
