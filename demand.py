import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data
print("Loading and preprocessing datasets...\n")
file_path2 = '/Users/giorgosziakas/Desktop/updated_data(2023-2022).csv'
df = pd.read_csv(file_path2, sep=';')
print(df.columns)

# Step 1: Parse date and time columns
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])  # Combine Date and Time
df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format

# Step 2: Aggregate Qty to daily level
daily_data = df.groupby('Date')['Qty'].sum().reset_index()

# Step 3: Plot the daily demand to visualize trends
plt.figure(figsize=(15, 6))
plt.plot(daily_data['Date'], daily_data['Qty'], label='Daily Demand')
plt.title('Daily Demand Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid()
plt.show()


# Step 4: Basic stats and missing values
print("Summary Statistics:")
print(daily_data.describe())

print("\nMissing Values:")
print(daily_data.isnull().sum())



# Aggregate to weekly demand
weekly_data = daily_data.groupby(pd.Grouper(key='Date', freq='W-SUN'))['Qty'].sum().reset_index()

# Rename columns for clarity
weekly_data.rename(columns={'Date': 'Week', 'Qty': 'Weekly Demand'}, inplace=True)

# Plot the weekly aggregated demand
plt.figure(figsize=(15, 6))
plt.plot(weekly_data['Week'], weekly_data['Weekly Demand'], marker='o', label='Weekly Demand')
plt.title('Weekly Demand Over Time')
plt.xlabel('Week')
plt.ylabel('Weekly Quantity')
plt.legend()
plt.grid()
plt.show()

# Print weekly demand stats
print("Weekly Demand Summary Statistics:")
print(weekly_data.describe())




from statsmodels.tsa.seasonal import seasonal_decompose

# Remove zero values after Sep 2024 (truncate dataset to non-zero weeks)
weekly_data = weekly_data[weekly_data['Weekly Demand'] > 0]

# Perform seasonal decomposition
result = seasonal_decompose(weekly_data['Weekly Demand'], model='additive', period=52)  # Assuming yearly seasonality

# Plot decomposition
result.plot()
plt.show()




from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Train-test split
train = weekly_data[:-12]  # Use all but the last 12 weeks for training
test = weekly_data[-12:]  # Use the last 12 weeks for testing

# Extract values for training
train_y = train['Weekly Demand'].values
test_y = test['Weekly Demand'].values

# Define SARIMA model (adjust parameters as needed)
sarima_model = SARIMAX(
    train_y,
    order=(1, 1, 1),          # (p, d, q) terms
    seasonal_order=(1, 1, 1, 52),  # (P, D, Q, S) terms (S = 52 for weekly seasonality)
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit the model
sarima_result = sarima_model.fit(disp=False)

# Print summary
print(sarima_result.summary())

# Display the selected parameters
print("Best SARIMA Parameters:", sarima_model.order, "Seasonal Order:", sarima_model.seasonal_order)

# Forecast for the next 12 weeks
forecast = sarima_result.forecast(steps=len(test_y))

# Evaluate model performance
mae = mean_absolute_error(test_y, forecast)
rmse = np.sqrt(mean_squared_error(test_y, forecast))
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(test['Week'], test_y, label='Actual Demand', marker='o')
plt.plot(test['Week'], forecast, label='Forecasted Demand', marker='o')
plt.title('SARIMA Model - Weekly Demand Forecast')
plt.xlabel('Week')
plt.ylabel('Demand')
plt.legend()
plt.grid()
plt.show()




# Correct the data handling to ensure forecast starts after actual data
forecast_weeks_9m = pd.date_range(start=weekly_data['Week'].iloc[-1] + pd.Timedelta(days=7), periods=44, freq='W-SUN')

# Forecast again for 9 months (36 weeks)
forecast_9_months = sarima_result.forecast(steps=44)

# Create a DataFrame for the forecast
forecast_df_9m = pd.DataFrame({'Week': forecast_weeks_9m, 'Weekly Demand': forecast_9_months})






# Combine actual and forecasted data for the testing period
test['Forecast'] = sarima_result.forecast(steps=len(test))

# Plot training, testing, and forecasted data
plt.figure(figsize=(15, 6))

# Plot training data
plt.plot(train['Week'], train['Weekly Demand'], label='Training Data', color='blue', marker='o')

# Plot testing data
plt.plot(test['Week'], test['Weekly Demand'], label='Testing Data (Actual)', color='green', marker='o')

# Plot forecasted data
plt.plot(test['Week'], test['Forecast'], label='Forecasted Data', linestyle='--', color='red', marker='o')

# Highlight forecast start
plt.axvline(x=test['Week'].iloc[0], color='gray', linestyle='--', label='Forecast Start')

# Add labels, title, and legend
plt.title('Training vs Testing Data with Forecast')
plt.xlabel('Week')
plt.ylabel('Weekly Demand')
plt.legend()
plt.grid()
plt.show()
