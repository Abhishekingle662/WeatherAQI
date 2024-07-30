from huggingface_hub import from_pretrained_keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_csv('data/daily_aqi_by_county_2020.csv')

# Select relevant columns and preprocess
data = data[['State Name', 'county Name', 'Date', 'AQI']]
data = data.dropna()

# Convert 'Date' to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Sample 10% of the data
data = data.sample(frac=0.10, random_state=42)

# Normalize 'AQI' values
scaler = MinMaxScaler(feature_range=(0, 1))
data['AQI_scaled'] = scaler.fit_transform(data[['AQI']])

# Encode categorical features
state_encoder = LabelEncoder()
county_encoder = LabelEncoder()

data['State Name'] = state_encoder.fit_transform(data['State Name'])
data['county Name'] = county_encoder.fit_transform(data['county Name'])

# Prepare the input features
features = data[['State Name', 'county Name', 'DayOfWeek', 'Month', 'AQI_scaled']].values

# We need to expand the features to 7 dimensions and sequence length to 120
num_features = 7
sequence_length = 120

# Expand features to 7 dimensions (fill with zeros)
features = np.pad(features, ((0, 0), (0, num_features - features.shape[1])), 'constant')

# Ensure there are enough samples to form sequences of length 120
num_samples = len(features)
if num_samples < sequence_length:
    print(f"Not enough data to form sequences of length {sequence_length}")
else:
    # Create sequences of length 120
    sequences = []
    targets = []
    for i in range(num_samples - sequence_length):
        sequences.append(features[i:i+sequence_length])
        targets.append(features[i+sequence_length, 4])  # AQI_scaled is at index 4
    
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Verify the shape
    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")

    # Create a simple LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, num_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(sequences, targets, epochs=10, batch_size=32, validation_split=0.2)

    # Make predictions
    predictions = model.predict(sequences)

    # Inverse transform the AQI predictions
    predicted_aqi = scaler.inverse_transform(predictions)

    # Display predictions
    print("\nPredicted AQI Statistics:")
    print(f"Mean: {np.mean(predicted_aqi)}")
    print(f"Min: {np.min(predicted_aqi)}")
    print(f"Max: {np.max(predicted_aqi)}")

    # Evaluate the first few predictions
    for i in range(10):
        print(f"Predicted AQI for sequence {i+1}: {predicted_aqi[i][0]:.2f}")

    # Compare with actual AQI values
    actual_aqi = data['AQI'].values[sequence_length:]
    print("\nActual AQI Statistics:")
    print(f"Mean: {np.mean(actual_aqi)}")
    print(f"Min: {np.min(actual_aqi)}")
    print(f"Max: {np.max(actual_aqi)}")

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(predicted_aqi.flatten() - actual_aqi))
    print(f"\nMean Absolute Error: {mae:.2f}")

    # Calculate Root Mean Squared Error
    rmse = np.sqrt(np.mean((predicted_aqi.flatten() - actual_aqi)**2))
    print(f"Root Mean Squared Error: {rmse:.2f}")