import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Streamlit app title
st.title("Stock Price Predictor App")

# Input for Indian stock symbol (e.g., 'RELIANCE.NS' for Reliance Industries)
stock = st.text_input("Enter the Stock ID (e.g., RELIANCE.NS)", "RELIANCE.NS")

# Define the date range for fetching data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Function to fetch stock data with rate limit handling
def fetch_stock_data(stock, start, end, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            data = yf.download(stock, start, end)
            if not data.empty:
                return data
            else:
                raise ValueError("Empty Data Received")
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                st.warning(f"Rate limit exceeded. Retrying in {5 * (retries + 1)} seconds...")
                time.sleep(5 * (retries + 1))  # Exponential backoff
                retries += 1
            else:
                st.error(f"Error fetching stock data: {e}")
                st.stop()
    st.error("Failed to fetch data after multiple attempts. Please try again later.")
    st.stop()

# Fetch stock data
google_data = fetch_stock_data(stock, start, end)

# Ensure model file exists before loading
model_path = "_Stock_Price_Prediction_model.keras"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Reset TensorFlow graph (to avoid warnings)
tf.compat.v1.reset_default_graph()

# Load the pre-trained model
model = load_model(model_path)

# Display the fetched stock data
st.subheader("Stock Data")
st.write(google_data)

# Splitting the data for testing
splitting_len = int(len(google_data) * 0.7)

# Fix: Ensure column names are properly referenced
if "Close" not in google_data.columns:
    st.error("Column 'Close' not found in fetched data. Check if the stock symbol is correct.")
    st.stop()

x_test = pd.DataFrame(google_data["Close"][splitting_len:])

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data["Close"], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Plot moving averages
for ma in [250, 200, 100]:
    st.subheader(f'Original Close Price and MA for {ma} days')
    google_data[f'MA_for_{ma}_days'] = google_data["Close"].rolling(ma).mean()
    st.pyplot(plot_graph((15, 6), google_data[f'MA_for_{ma}_days'], google_data, 0))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data = []
y_data = []

# Prepare the data for prediction
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse transform the predictions
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare data for plotting
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# Display original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot original vs predicted close prices
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data["Close"][:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)





