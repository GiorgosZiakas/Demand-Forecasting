import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load and Preprocess Data
file_path_2023 = '/Users/giorgosziakas/Desktop/data(2023-2022).csv'
file_path_2024 = '/Users/giorgosziakas/Desktop/jan-nov.csv'

# Load both datasets
df_2023_2022 = pd.read_csv(file_path_2023, sep=';')
df_2024 = pd.read_csv(file_path_2024, sep=';')

# Combine the datasets
df_combined = pd.concat([df_2023_2022, df_2024])
# Parse date column with mixed formats

df_combined['Date'] = pd.to_datetime(df_combined['Date'], format='mixed', dayfirst=True)

# Aggregate daily data
daily_data = df_combined.groupby('Date')['Qty'].sum().reset_index()

# Print the first few rows to ensure the data looks correct
print(daily_data.head())

# Continue with your analysis or modeling steps

# Step 2 (EDA): Visualize Daily Demand (optional)
# Uncomment the following lines for EDA
# plt.figure(figsize=(15, 6))
# plt.plot(daily_data['Date'], daily_data['Qty'], label='Daily Demand')
# plt.title('Daily Demand Over Time')
# plt.xlabel('Date')
# plt.ylabel('Quantity')
# plt.legend()
# plt.grid()
# plt.show()

# print("Daily Data Summary Statistics:")
# print(daily_data.describe())
# print("\nMissing Values in Daily Data:")
# print(daily_data.isnull().sum())

# Step 3: Aggregate to Weekly Demand
weekly_data = daily_data.groupby(pd.Grouper(key='Date', freq='W-SUN'))['Qty'].sum().reset_index()
weekly_data.rename(columns={'Date': 'Week', 'Qty': 'Weekly Demand'}, inplace=True)

# Step 4 (EDA): Visualize Weekly Demand (optional)
# Uncomment the following lines for EDA
# plt.figure(figsize=(15, 6))
# plt.plot(weekly_data['Week'], weekly_data['Weekly Demand'], marker='o', label='Weekly Demand')
# plt.title('Weekly Demand Over Time')
# plt.xlabel('Week')
# plt.ylabel('Weekly Quantity')
# plt.legend()
# plt.grid()
# plt.show()

# print("Weekly Data Summary Statistics:")
# print(weekly_data.describe())

# Step 5 (EDA): Seasonal Decomposition (optional)
# Uncomment the following lines for EDA
# result = seasonal_decompose(weekly_data['Weekly Demand'], model='additive', period=52)
# result.plot()
# plt.show()

# Remove zero values for SARIMA compatibility
weekly_data = weekly_data[weekly_data['Weekly Demand'] > 0]

# Step 6: Train-Test Split
train = weekly_data[:-12]  # Use all but the last 12 weeks for training
test = weekly_data[-12:]  # Use the last 12 weeks for testing
train_y = train['Weekly Demand'].values
test_y = test['Weekly Demand'].values

# Step 7: Train Final SARIMA Model with Best Parameters
best_params = {'p': 1, 'd': 1, 'q': 3, 'P': 1, 'D': 0, 'Q': 0}
sarima_model = SARIMAX(
    train_y,
    order=(best_params['p'], best_params['d'], best_params['q']),
    seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_result = sarima_model.fit(disp=False)

# Step 8: Evaluate the Final Model
forecast = sarima_result.forecast(steps=len(test_y))
mae = mean_absolute_error(test_y, forecast)
rmse = np.sqrt(mean_squared_error(test_y, forecast))
print(f"Final Model MAE: {mae}")
print(f"Final Model RMSE: {rmse}")

# Visualize Training vs Testing Data
plt.figure(figsize=(15, 6))
plt.plot(train['Week'], train['Weekly Demand'], label='Training Data', color='blue', marker='o')
plt.plot(test['Week'], test['Weekly Demand'], label='Testing Data (Actual)', color='green', marker='o')
plt.plot(test['Week'], forecast, label='Forecasted Data', linestyle='--', color='red', marker='o')
plt.axvline(x=test['Week'].iloc[0], color='gray', linestyle='--', label='Forecast Start')
plt.title('Final SARIMA Model - Training vs Testing with Forecast')
plt.xlabel('Week')
plt.ylabel('Weekly Demand')
plt.legend()
plt.grid()
plt.show()

# Step 9: Forecast Next 9 Months (36 Weeks)
forecast_9_months = sarima_result.forecast(steps=16)
forecast_weeks_9m = pd.date_range(start=weekly_data['Week'].iloc[-1] + pd.Timedelta(days=7), periods=16, freq='W-SUN')
forecast_df_9m = pd.DataFrame({'Week': forecast_weeks_9m, 'Weekly Demand': forecast_9_months})

# Visualize Actual + 9-Month Forecast
plt.figure(figsize=(15, 6))
plt.plot(weekly_data['Week'], weekly_data['Weekly Demand'], label='Actual Weekly Demand', marker='o')
plt.plot(forecast_df_9m['Week'], forecast_df_9m['Weekly Demand'], label='Forecasted Demand (Next 9 Months)', linestyle='--', color='red', marker='o')
plt.axvline(x=test['Week'].iloc[-1], color='gray', linestyle='--', label='Forecast Start')

# Format x-axis to display months
ax = plt.gca()  # Get the current axis
ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set locator to display each month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.title('Weekly Demand with 9-Month Forecast Continuation')
plt.xlabel('Week')
plt.ylabel('Weekly Demand')
plt.legend()
plt.grid()
plt.show()


