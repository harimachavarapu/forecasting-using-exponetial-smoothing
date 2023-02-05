# forecasting-using-exponetial-smoothing
Optimization of Supply Chain Management 

business problem - Logistic company is facing lots of losses due to irregular demand of their sales

business objective - to reduce loss

A holt winter expontential smoothing model is build to predict the future sales so that the cilent can avoid looses
As part of EDA is used basic excel and combined data into months by adding all days in that month.
After doing that I taken that data and started building model.

#MODEL BUILDING:

The model is built in Python and uses the following libraries: Pandas, Numpy, Statsmodels, and Streamlit.
The purpose of the model is to perform time series forecasting using the Exponential Smoothing method.
The data is splited into train test split
The code defines the Exponential Smoothing model using the statsmodels library's ExponentialSmoothing class. The model is specified with a "trend" component and a "seasonal" component
Fit the model: The model is fit to the train data using the fit method.
Number of steps to forecast: The user is asked to specify the number of steps to forecast using a Streamlit slider widget.
Forecast button: If the user clicks the "Forecast" button, the code uses the fitted model to forecast the next steps number of observations.
Plotting the forecast: The forecast is plotted using Streamlit's line_chart function.
Evaluation: The code calculates the root mean squared error (RMSE) between the forecast and the actual test data. The RMSE is a measure of the accuracy of the forecast.
