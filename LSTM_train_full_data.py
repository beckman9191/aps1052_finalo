import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, AdditiveAttention, Permute, Reshape, Multiply
from joblib import dump
from function import create_model

company = 'TSLA'
data = pd.read_csv(f'data/{company}.csv')
timestep = 60

# Convert 'Date' column to datetime format if it's not already
data['Date'] = pd.to_datetime(data['Date'])


# Split the data into training and test sets
train_data = data


# Load and Preprocessing Training Data
train_data["Close"] = pd.to_numeric(train_data.Close, errors='coerce')
train_data = train_data.dropna()
print(train_data['Date'])
train_data = train_data.iloc[:, 4:5].values


#Scaling the Data for NN
sc = MinMaxScaler(feature_range=(0, 1))
train_data = sc.fit_transform(train_data)

# Save the fitted scaler
dump(sc, f'models/LSTM_scaler_full_data_{company}.joblib')


X_train = []
y_train = []

for i in range(timestep, len(train_data)):
    X_train.append(train_data[i-timestep: i, 0])
    y_train.append(train_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #adding the batch_size


#model = create_model=(timestep)
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

# train
hist = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

# Save the model to a file
#model.save('models/LSTM_TSLA_model')

# Save only the weights of the model
#model.save_weights('models/LSTM_TSLA_weights.h5')

# Saving weights manually to avoid potential meta-data issues
weights = model.get_weights()  # Gets the weights and biases in a list of numpy arrays.
np.savez(f'models/LSTM_weights_full_data_{company}.npz', *weights)

plt.plot(hist.history['loss'])
plt.title('Training Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(f'Results/training_loss_full_data_{company}.png')

plt.show()




