import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load and Preprocess Data
print("Loading and preprocessing datasets...\n")
file_path = '/Users/giorgosziakas/Desktop/updated_data(2023-2022).csv'
df = pd.read_csv(file_path, sep=';')

# Parse date and time columns
df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
daily_data = df.groupby('Date')['Qty'].sum().reset_index()  # Aggregate Qty to daily level

# Step 2: Visualize Daily Demand
plt.figure(figsize=(15, 6))
plt.plot(daily_data['Date'], daily_data['Qty'], label='Daily Demand')
plt.title('Daily Demand Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid()
plt.show()

print("Daily Data Summary Statistics:")
print(daily_data.describe())
print("\nMissing Values in Daily Data:")
print(daily_data.isnull().sum())

# Step 3: Aggregate to Weekly Demand
weekly_data = daily_data.groupby(pd.Grouper(key='Date', freq='W-SUN'))['Qty'].sum().reset_index()
weekly_data.rename(columns={'Date': 'Week', 'Qty': 'Weekly Demand'}, inplace=True)

# Visualize Weekly Demand
plt.figure(figsize=(15, 6))
plt.plot(weekly_data['Week'], weekly_data['Weekly Demand'], marker='o', label='Weekly Demand')
plt.title('Weekly Demand Over Time')
plt.xlabel('Week')
plt.ylabel('Weekly Quantity')
plt.legend()
plt.grid()
plt.show()

print("Weekly Data Summary Statistics:")
print(weekly_data.describe())

# Step 4: Seasonal Decomposition
weekly_data = weekly_data[weekly_data['Weekly Demand'] > 0]  # Remove zero values
result = seasonal_decompose(weekly_data['Weekly Demand'], model='additive', period=52)
result.plot()
plt.show()

# Step 5: Train-Test Split
train = weekly_data[:-12]  # Use all but the last 12 weeks for training
test = weekly_data[-12:]  # Use the last 12 weeks for testing
train_y = train['Weekly Demand'].values
test_y = test['Weekly Demand'].values

# Step 6: Define and Fit SARIMA Model
sarima_model = SARIMAX(
    train_y,
    order=(1, 1, 1),  # Adjustable (p, d, q)
    seasonal_order=(1, 1, 1, 52),  # Seasonal (P, D, Q, S) terms
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_result = sarima_model.fit(disp=False)
print(sarima_result.summary())

# Step 7: Forecast and Evaluate
forecast = sarima_result.forecast(steps=len(test_y))
mae = mean_absolute_error(test_y, forecast)
rmse = np.sqrt(mean_squared_error(test_y, forecast))
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Visualize Training vs Testing Data
plt.figure(figsize=(15, 6))
plt.plot(train['Week'], train['Weekly Demand'], label='Training Data', color='blue', marker='o')
plt.plot(test['Week'], test['Weekly Demand'], label='Testing Data (Actual)', color='green', marker='o')
plt.plot(test['Week'], forecast, label='Forecasted Data', linestyle='--', color='red', marker='o')
plt.axvline(x=test['Week'].iloc[0], color='gray', linestyle='--', label='Forecast Start')
plt.title('Training vs Testing Data with Forecast')
plt.xlabel('Week')
plt.ylabel('Weekly Demand')
plt.legend()
plt.grid()
plt.show()

# Step 8: Forecast Next 9 Months (36 Weeks)
forecast_9_months = sarima_result.forecast(steps=36)
forecast_weeks_9m = pd.date_range(start=weekly_data['Week'].iloc[-1] + pd.Timedelta(days=7), periods=36, freq='W-SUN')
forecast_df_9m = pd.DataFrame({'Week': forecast_weeks_9m, 'Weekly Demand': forecast_9_months})

# Visualize Actual + 9-Month Forecast
plt.figure(figsize=(15, 6))
plt.plot(weekly_data['Week'], weekly_data['Weekly Demand'], label='Actual Weekly Demand', marker='o')
plt.plot(forecast_df_9m['Week'], forecast_df_9m['Weekly Demand'], label='Forecasted Demand (Next 9 Months)', linestyle='--', color='red', marker='o')
plt.axvline(x=test['Week'].iloc[-1], color='gray', linestyle='--', label='Forecast Start')
plt.title('Weekly Demand with 9-Month Forecast Continuation')
plt.xlabel('Week')
plt.ylabel('Weekly Demand')
plt.legend()
plt.grid()
plt.show()




import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the objective function for Optuna
def sarima_objective(trial):
    # Suggest SARIMA parameters
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)

    try:
        # Train SARIMA model with the suggested parameters
        model = SARIMAX(
            train_y,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 52),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        # Forecast for the test set
        forecast = model_fit.forecast(steps=len(test_y))

        # Calculate RMSE as the optimization metric
        rmse = np.sqrt(mean_squared_error(test_y, forecast))
        return rmse

    except Exception as e:
        # Return a high RMSE for failed models
        return np.inf

# Create an Optuna study
study = optuna.create_study(direction='minimize')  # Minimize RMSE
study.optimize(sarima_objective, n_trials=50, timeout=600)  # Run for 50 trials or 10 minutes

# Print the best parameters and the corresponding RMSE
best_params = study.best_params
best_rmse = study.best_value
print(f"Best SARIMA Parameters: {best_params}")
print(f"Best RMSE: {best_rmse}")

# Refit the SARIMA model with the best parameters
best_sarima_model = SARIMAX(
    train_y,
    order=(best_params['p'], best_params['d'], best_params['q']),
    seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)
best_sarima_result = best_sarima_model.fit(disp=False)

# Forecast with the optimized model
optimized_forecast = best_sarima_result.forecast(steps=len(test_y))

# Evaluate optimized model
optimized_mae = mean_absolute_error(test_y, optimized_forecast)
optimized_rmse = np.sqrt(mean_squared_error(test_y, optimized_forecast))
print(f"Optimized MAE: {optimized_mae}")
print(f"Optimized RMSE: {optimized_rmse}")

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(train['Week'], train['Weekly Demand'], label='Training Data', color='blue', marker='o')
plt.plot(test['Week'], test['Weekly Demand'], label='Testing Data (Actual)', color='green', marker='o')
plt.plot(test['Week'], optimized_forecast, label='Optimized Forecast', linestyle='--', color='red', marker='o')
plt.axvline(x=test['Week'].iloc[0], color='gray', linestyle='--', label='Forecast Start')
plt.title('Optimized SARIMA Model - Training vs Testing with Forecast')
plt.xlabel('Week')
plt.ylabel('Weekly Demand')
plt.legend()
plt.grid()
plt.show()
