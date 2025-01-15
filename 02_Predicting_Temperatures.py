import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the data
file_path = 'data\weather_data_goa_hourly.csv'
weather_data = pd.read_csv(file_path)

# Convert `date_time` to datetime object
weather_data['date_time'] = pd.to_datetime(weather_data['date_time'])

# Extract relevant features
weather_data['hour'] = weather_data['date_time'].dt.hour
weather_data['day_of_week'] = weather_data['date_time'].dt.dayofweek

# Create lag features for temperature
weather_data['tempC_lag1'] = weather_data['tempC'].shift(1)
weather_data.dropna(inplace=True)

# Select features and normalize
features = ['tempC', 'humidity', 'pressure', 'precipMM', 'uvIndex', 'windspeedKmph', 'winddirDegree', 'tempC_lag1', 'hour', 'day_of_week']
scaler = MinMaxScaler()
weather_data[features] = scaler.fit_transform(weather_data[features])

# Prepare the data for LSTM
sequence_length = 24  # Use the last 24 hours to predict the next hour
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][0]  # Predict the temperature (or any other target)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Split into sequences
X, y = create_sequences(weather_data[features].values, sequence_length)

# Split the data into training and validation sets
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Update the future_dates range to cover only from 16th January 2025 to the end of January 2025
future_dates = pd.date_range(start='2025-01-16', end='2025-01-31', freq='H')
predictions = []

# Use the last sequence from the dataset for prediction
last_sequence = weather_data[features].values[-sequence_length:]

for date in future_dates:
    
    print(f"Predicting for {date}")

    # Predict the next temperature using the LSTM model
    next_prediction = model.predict(last_sequence[np.newaxis, :, :])
    
    # Reshape next_prediction to ensure it is a 1D array
    next_prediction = next_prediction.flatten()
    
    if np.isnan(next_prediction).any():
        print(f"NaN encountered at {date}, stopping prediction.")
        break

    # Update the last sequence with the new prediction
    last_sequence = np.vstack([last_sequence[1:], np.hstack([next_prediction, last_sequence[-1, 1:]])])
    
    # Append the prediction for the specific date
    predictions.append(next_prediction)

# Check if predictions are non-empty before proceeding
if predictions:
    # Inverse transform the predictions to the original scale
    predictions_array = np.array(predictions).reshape(-1, 1)
    transformed_predictions = scaler.inverse_transform(np.hstack([predictions_array, np.zeros((len(predictions_array), len(features)-1))]))[:, 0]

    # Prepare the results
    df_predictions = pd.DataFrame({'date_time': future_dates, 'predicted_tempC': transformed_predictions})
    print(df_predictions.head())
else:
    print("No predictions were made.")


# Plotting Graphs

import matplotlib.pyplot as plt

plt.plot(df_predictions['date_time'], df_predictions['predicted_tempC'])
plt.title('Predicted Temperatures from January 16 to 31, 2025')
plt.xlabel('Date Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.show()
