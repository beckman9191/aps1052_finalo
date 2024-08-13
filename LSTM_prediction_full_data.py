import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from joblib import load
from datetime import datetime, timedelta
from function import create_model


company = 'TSLA'
timestep = 60
period = 14

# Load the saved model and scaler
sc = load(f'models/LSTM_scaler_full_data_{company}.joblib')
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
with np.load(f'models/LSTM_weights_full_data_{company}.npz') as data:
    loaded_weights = [data[key] for key in sorted(data.keys(), key=lambda x: int(x.split('_')[1]))]
model.set_weights(loaded_weights)

# Compile the model for making predictions
model.compile(optimizer='adam', loss="mean_squared_error")


# Read the data
data = pd.read_csv(f'data/{company}.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime format if it's not already

# Extract the last {timestep} days data
latest_data = data.iloc[-timestep:, 4:5]  # Assuming 'Close' is the 5th column
print(latest_data)
# Prepare data for prediction
latest_data_scaled = sc.transform(latest_data)
X_latest = np.reshape(latest_data_scaled, (1, timestep, 1))  # Reshape for LSTM

# Predict the next {period} business days prices
predicted_prices = []
current_input = X_latest
for i in range(period):
    # Predict the next day price
    next_price_scaled = model.predict(current_input)
    print(sc.inverse_transform(next_price_scaled))
    predicted_prices.append(next_price_scaled[0, 0])  # Store the scaled prediction
    
    # Update current_input for next prediction
    current_input = np.append(current_input[:, 1:, :], next_price_scaled.reshape(1, 1, 1), axis=1)
    if i == 1:
        print(sc.inverse_transform(current_input.reshape(-1, 1)))

# Transform predicted prices back to original scale
predicted_prices_array = np.array(predicted_prices).reshape(-1, 1)
predicted_prices_inversed = sc.inverse_transform(predicted_prices_array)

# Generate future business dates
last_date = data['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=period + 1, freq='B')[1:]  # Use 'B' for business day frequency

# Create DataFrame to save to CSV
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': predicted_prices_inversed.flatten()
})

# Save DataFrame to CSV
predictions_df.to_csv(f'Prediction/predicted_prices_full_data_{company}.csv', index=False)


# Ensure future_dates and predicted_prices_inversed are aligned
print(f"Dates length: {len(future_dates)}, Prices length: {len(predicted_prices_inversed)}")

fig = plt.figure(figsize=(15, 7))
plt.plot(future_dates, predicted_prices_inversed.flatten(), label='Predicted Prices')
plt.title(f'{company} Stock Price Forecast for Next {period} Business Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

#save the plot
fig.savefig(f'Prediction/{period}_day_forecast_full_data_{company}.png')
plt.show()
