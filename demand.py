import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import optuna

# Step 1: Load and Preprocess Data
print("Loading and preprocessing datasets...\n")
file_path1 = '/Users/giorgosziakas/Desktop/D2024.csv'
file_path2 = '/Users/giorgosziakas/Desktop/data(2023-2022).csv'
data1 = pd.read_csv(file_path1, sep=';')
data2 = pd.read_csv(file_path2, sep=';')
data = pd.concat([data1, data2], ignore_index=True)

# Convert 'Date' to datetime and aggregate daily data
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%y', errors='coerce')
data = data[['Date', 'Qty']].groupby('Date').sum().reset_index()

# Remove negative values in Qty
data = data[data['Qty'] > 0]

# Winsorize extreme values (remove outliers)
print("Clipping extreme values...\n")
lower_limit = data['Qty'].quantile(0.05)
upper_limit = data['Qty'].quantile(0.95)
data['Qty'] = data['Qty'].clip(lower=lower_limit, upper=upper_limit)

# Aggregate weekly demand
print("Aggregating data to weekly level...\n")
data.set_index('Date', inplace=True)
weekly_data = data.resample('W').sum().reset_index()

# Add exogenous features
print("Adding exogenous features...\n")
weekly_data['is_weekend'] = weekly_data['Date'].dt.dayofweek.isin([5, 6]).astype(int)
weekly_data['rolling_avg_7day'] = weekly_data['Qty'].rolling(window=4, min_periods=1).mean()
weekly_data['is_december'] = (weekly_data['Date'].dt.month == 12).astype(int)

# Train-test split
train_size = int(len(weekly_data) * 0.8)
train_data = weekly_data.iloc[:train_size]
test_data = weekly_data.iloc[train_size:]

# Step 2: Define the Objective Function for Bayesian Optimization
def objective(trial):
    # Suggest SARIMA parameters
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)
    s = 52  # Fixed seasonal period (yearly seasonality for weekly data)

    try:
        # Fit SARIMA model with exogenous variables
        model = SARIMAX(
            train_data['Qty'],
            exog=train_data[['is_weekend', 'rolling_avg_7day', 'is_december']],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        return results.aic  # Use AIC as the objective to minimize
    except Exception:
        return float('inf')  # Return a high value if the model fails

# Step 3: Run Bayesian Optimization
print("Starting Bayesian Optimization for SARIMA parameters...\n")
study = optuna.create_study(direction='minimize')  # Minimize AIC
study.optimize(objective, n_trials=50)  # Run 50 trials

# Retrieve the best parameters
best_params = study.best_params
print(f"Best SARIMA Parameters: {best_params}")

# Step 4: Fit the Best SARIMA Model
print("Fitting the best SARIMA model...\n")
sarima_model = SARIMAX(
    train_data['Qty'],
    exog=train_data[['is_weekend', 'rolling_avg_7day', 'is_december']],
    order=(best_params['p'], best_params['d'], best_params['q']),
    seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_results = sarima_model.fit(disp=False)
print(sarima_results.summary())

# Step 5: Forecast on Test Set
print("Forecasting for Test Set and Future (6 Months)...\n")

# Generate future dates for 6 months (26 weeks)
future_dates = pd.date_range(start=weekly_data['Date'].iloc[-1] + pd.Timedelta(weeks=1), periods=39, freq='W')

# Create future exogenous features
future_exog = pd.DataFrame({
    'Date': future_dates,
    'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int),
    'rolling_avg_7day': test_data['rolling_avg_7day'].iloc[-1],  # Use last known rolling average
    'is_december': (future_dates.month == 12).astype(int)
})

# Forecast both test set and extended future
forecast = sarima_results.get_forecast(
    steps=len(test_data) + len(future_dates),
    exog=pd.concat([test_data[['is_weekend', 'rolling_avg_7day', 'is_december']], 
                    future_exog[['is_weekend', 'rolling_avg_7day', 'is_december']]])
)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Add forecast for future data
future_forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Qty': forecast_mean[-39:].values  # Extract future predictions
})

# Step 6: Evaluate Model Performance
print("Evaluating model performance...\n")
test_rmse = np.sqrt(mean_squared_error(test_data['Qty'], forecast_mean[:len(test_data)]))
print(f"Test Set RMSE: {test_rmse:.4f}")

# Step 7: Plot Actual, Test, and Future Forecasts
plt.figure(figsize=(14, 8))
plt.plot(train_data['Date'], train_data['Qty'], label="Training Data", color="blue")
plt.plot(test_data['Date'], test_data['Qty'], label="Actual Test Data", color="green")
plt.plot(test_data['Date'], forecast_mean[:len(test_data)], label="Forecasted Test Data", color="orange")
plt.plot(future_forecast['Date'], future_forecast['Predicted_Qty'], label="Future Forecast (6 Months)", color="purple")

# Combine dates for confidence interval plotting
combined_dates = pd.concat([test_data['Date'], future_forecast['Date']])
plt.fill_between(
    combined_dates,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="orange",
    alpha=0.2,
    label="Confidence Interval"
)

plt.axvline(x=test_data['Date'].iloc[0], color="red", linestyle="--", label="Train-Test Split")
plt.title("SARIMA Forecast with Bayesian Optimization (6 Months Extended Horizon)")
plt.xlabel("Date")
plt.ylabel("Demand (Qty)")
plt.legend()
plt.grid(True)
plt.show()

