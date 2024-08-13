import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from joblib import load
from function import create_model

company = 'META'
timestep = 180

# Load the saved model and scaler
sc = load(f'models/LSTM_scaler_{company}.joblib')
#sc = load(f'models/preprocessing_pipeline_{company}.joblib')
# Recreate the model architecture
model = create_model(timestep)

# Loading weights
with np.load(f'models/LSTM_weights_{company}.npz') as data:
    loaded_weights = [data[key] for key in sorted(data.keys(), key=lambda x: int(x.split('_')[1]))]
model.set_weights(loaded_weights)


# Compile the model again if you plan to continue training or make predictions
model.compile(optimizer='adam', loss="mean_squared_error")


# read the data
data = pd.read_csv(f'data/{company}.csv')

# Convert 'Date' column to datetime format if it's not already
data['Date'] = pd.to_datetime(data['Date'])

# Set the date for the last 2 years
last_year_date = data['Date'].max() - pd.DateOffset(years=2)
# Split into training and test sets (80% train, 20% test)
test_data = data[data['Date'] >= last_year_date]

test_data['Close'] = pd.to_numeric(test_data.Close, errors='coerce')
test_data = test_data.dropna()
test_data = test_data.iloc[:, 4:5]

y_test = test_data.iloc[timestep:,0:].values
#input array for the model
inputClosing = test_data.iloc[:,0:].values
#print(inputClosing[-timestep:])
inputClosing_scaled = sc.transform(inputClosing)

X_test = []
length = len(test_data)


for i in range(timestep, length):
    X_test.append(inputClosing_scaled[i-timestep:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_test.shape)

# Using previous predictions for future predictions
predicted_price = []

for i in range(len(X_test)):
    pred = model.predict(X_test[i].reshape(1, timestep, 1))
    predicted_price.append(pred[0, 0])
    
    # Use the predicted value for the next prediction
    if i < len(X_test) - 1:
        X_test[i + 1, :-1, 0] = X_test[i + 1, 1:, 0]
        X_test[i + 1, -1, 0] = pred

predicted_price = np.array(predicted_price).reshape(-1, 1)
predicted_price = sc.inverse_transform(predicted_price)


#y_pred = model.predict(X_test)
#print(y_pred)
#predicted_price = sc.inverse_transform(y_pred)
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

