
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import streamlit as st
from PIL import Image

st.title("Time Series Forecasting By - HK")

#LOADING IMAGE
image = Image.open('C:/Users/hudso/Downloads/FB_IMG_1653804268741.jpg')
st.image(image, caption='HARI KRISHNA SRI SAI PRASAD MACHAVARAPU')
st.snow()#ADDS SNOW EFFECTS
st.balloons()#ADD BALLONS

# Load the data set
df = pd.read_csv("C:/Users/hudso/Downloads/fedex/plasticallot.csv")

# Convert the date column to a datetime type
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Set the date column as the index
df = df.set_index('date')

# Plot the data set
st.line_chart(df)

# Train/Test split
train_data = df[:-12]
test_data = df[-12:]

# Define the model
model = ExponentialSmoothing(train_data, trend="add", seasonal="add", seasonal_periods=12)

# Fit the model
model_fit = model.fit()

# Number of steps to forecast (slider)
steps = st.slider("Number of steps to forecast", min_value=1, max_value=12, value=12)

# Forecast button
if st.button("Forecast"):
    # Forecast for next steps
    forecast = model_fit.forecast(steps=steps)

    # Convert the forecast to a DataFrame
    forecast = pd.DataFrame(forecast, columns=["Forecast"], index=test_data.index[:steps])

    # Plot the forecast
    st.line_chart(forecast)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(test_data[:steps], forecast))
    st.write("RMSE: ", rmse)

    # Display the forecast in a table
    st.dataframe(forecast)
