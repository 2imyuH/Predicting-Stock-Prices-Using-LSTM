import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path="data/sample.csv"):
    df = pd.read_csv(path)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.sort_values('Date/Time')
    return df

def preprocess_data(df, feature_col='Close', look_back=60):
    """
    Chuẩn hóa dữ liệu và tạo sequences cho LSTM.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df[[feature_col]])

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape để phù hợp input LSTM: [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler
