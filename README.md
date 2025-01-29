import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset (make sure you have the 'AAPL_stock_data.csv' file in your directory)
df = pd.read_csv('AAPL_stock_data.csv')  # Replace with your file path

# Display the first few rows of the dataset
print(df.head())

# Data Preprocessing
# Use only 'Date' and 'Close' columns for prediction

df = df[['Date', 'Close']]

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print(df)
# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Fill missing values (if any)
df = df.bfill()


# Plot the closing prices
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Stock Price History')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Create training and test datasets (80% for training, 20% for testing)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create sequences for Random Forest
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create sequences for training and testing
X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Reshaping the data for Random Forest
X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten the features
X_test = X_test.reshape(X_test.shape[0], -1)

# Build the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Invert the scaling to get actual stock prices
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Stock Price')
plt.plot(predictions, label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# RMSE (Root Mean Squared Error) evaluation
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'RMSE: {rmse}')
