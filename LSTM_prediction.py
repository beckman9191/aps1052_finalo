import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from joblib import load
from datetime import datetime, timedelta

timestep = 180
company = 'LTO'

# Load the saved model and scaler
sc = load(f'models/LSTM_scaler_{company}.joblib')
print("Min:", sc.data_min_)
print("Max:", sc.data_max_)
print("Scale:", sc.data_max_ - sc.data_min_)
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

# Load and preprocess training data
data = pd.read_csv(f'data/{company}.csv')
data["Close"] = pd.to_numeric(data.Close, errors='coerce')
data = data.dropna()



# Handle dates for plotting and future prediction
last_date = pd.to_datetime(data['Date'].iloc[-1])

# Prepare the initial input (last 120 days)
input_values = data["Close"].values[-timestep:]  # Extract the last 120 days' Close prices
input_values = input_values.reshape(-1, 1)
print(input_values)
input_values_scaled = sc.transform(input_values)  # Scale the values and reshape for transform
input_seq = input_values_scaled.reshape(1, timestep, 1)  # Reshape for model input



predicted_prices = []
current_input = input_seq

# Iteratively predict the next 90 days
for _ in range(90):
    # Predict the next step
    next_price = model.predict(current_input)
    
    predicted_prices.append(sc.inverse_transform(next_price))  # Inverse transform to get the actual price

    # Update the current_input to include the predicted price and drop the oldest price
    current_input = np.append(current_input[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)

# Flatten the list of predictions to match the future_dates length
predicted_prices = np.array(predicted_prices).flatten()

# Generate Future Dates
future_dates = [last_date + timedelta(days=i) for i in range(1, 91)]  # Next 90 days

print(len(future_dates))  # Should be 90
print(len(predicted_prices.flatten()))  # Should also be 90

# Create a DataFrame with Future Dates and Predicted Prices
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': predicted_prices.flatten()
})

# Save to CSV
predicted_df.to_csv(f'Prediction/3_month_price_prediction_{company}.csv', index=False)

# Plot the Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(predicted_df['Date'], predicted_df['Predicted_Price'], color='green', label='Predicted 3-Month Stock Price')
plt.title(f'{company} 3-Month Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)

plt.savefig(f'Prediction/3_month_price_prediction_{company}.png')
plt.show()
