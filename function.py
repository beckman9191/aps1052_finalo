import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


class CustomPCA(PCA):
    def __init__(self, n_components=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        
def create_lstm_dataset(data, timestep):
    X, y = [], []
    for i in range(timestep, len(data)):
        X.append(data[i-timestep:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, y

def create_model(timestep):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(timestep, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss="mean_squared_error")
    return model