import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from joblib import load

company = 'LTO'
timestep = 180

# Load the saved model and scaler
sc = load(f'models/LSTM_scaler_{company}.joblib')
# Recreate the model architecture
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

# Loading weights
with np.load(f'models/LSTM_{company}_weights.npz') as data:
    loaded_weights = [data[key] for key in sorted(data.keys(), key=lambda x: int(x.split('_')[1]))]
model.set_weights(loaded_weights)

# Compile the model again if you plan to continue training or make predictions
model.compile(optimizer='adam', loss="mean_squared_error")

testData = pd.read_csv(f'data/test_{company}.csv')
testData['Close'] = pd.to_numeric(testData.Close, errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:, 4:5]

y_test = testData.iloc[timestep:,0:].values
#input array for the model
inputClosing = testData.iloc[:,0:].values
#print(inputClosing[-timestep:])
inputClosing_scaled = sc.transform(inputClosing)

X_test = []
length = len(testData)


for i in range(timestep, length):
    X_test.append(inputClosing_scaled[i-timestep:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


y_pred = model.predict(X_test)
#print(y_pred)

predicted_price = sc.inverse_transform(y_pred)
#print(predicted_price)

plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title(f'{company} stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

# Save the plot as a PNG file
plt.savefig(f'Results/price_prediction_{company}.png')

plt.show()

