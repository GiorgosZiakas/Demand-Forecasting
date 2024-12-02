import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np 

# Load the data
file_path1 = '/Users/giorgosziakas/Desktop/D2024.csv'
file_path2 = '/Users/giorgosziakas/Desktop/data(2023-2022).csv'
data1 = pd.read_csv(file_path1, sep=';')
data2 = pd.read_csv(file_path2, sep=';')

# Merge the two datasets
data = pd.concat([data1, data2], ignore_index=True)

def clean_data(data):
    """
    Cleans the raw dataset:
    - Converts Date to datetime.
    - Selects and renames relevant columns.
    - Cleans and standardizes text columns.
    """
    # Convert 'Date' to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%y', errors='coerce')
    
    # Filter relevant columns
    relevant_columns = ['Date', 'Time', 'Category', 'Item', 'Qty']
    cleaned_data = data[relevant_columns].copy()
    
    # Standardize text columns (strip and lowercase)
    cleaned_data['Category'] = cleaned_data['Category'].str.strip().str.lower()
    cleaned_data['Item'] = cleaned_data['Item'].str.strip().str.lower()
    
    return cleaned_data

def perform_eda(data):
    """
    Performs Exploratory Data Analysis:
    - Aggregates data by day, week, and hour.
    - Analyzes demand trends by Category and Item.
    Returns key summary statistics and plots.
    """
    # Create day_of_week column
    data['day_of_week'] = data['Date'].dt.day_name()
    
    # Daily demand
    daily_demand = data.groupby('Date')['Qty'].sum()
    
    # Weekly demand (by day of week)
    weekly_demand = data.groupby('day_of_week')['Qty'].sum()
    
    # Hourly demand
    data['hour'] = pd.to_datetime(data['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    hourly_demand = data.groupby('hour')['Qty'].sum()
    
    # Category and Item demand
    category_demand = data.groupby('Category')['Qty'].sum().sort_values(ascending=False)
    item_demand = data.groupby('Item')['Qty'].sum().sort_values(ascending=False)
    
    return {
        'daily_demand': daily_demand,
        'weekly_demand': weekly_demand,
        'hourly_demand': hourly_demand,
        'category_demand': category_demand,
        'item_demand': item_demand
    }

def feature_engineering(data, top_categories, top_items):
    """
    Performs feature engineering:
    - Creates time-based features (hour, day_of_week, is_weekend).
    - Adds rolling average (7-day) for historical demand trends.
    - Encodes top categories and items.
    """
    # Create day_of_week column (if not already present)
    if 'day_of_week' not in data.columns:
        data['day_of_week'] = data['Date'].dt.day_name()
    
    # Create time-based features
    data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    data['month'] = data['Date'].dt.month
    data['week_of_year'] = data['Date'].dt.isocalendar().week
    data['hour'] = pd.to_datetime(data['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Rolling average (7-day) for each item
    data['rolling_avg_7day'] = data.groupby('Item')['Qty'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Encode top categories
    for category in top_categories:
        data[f'is_{category}'] = (data['Category'] == category).astype(int)
    
    # Encode top items
    data['top_item'] = data['Item'].apply(lambda x: x if x in top_items else 'other')
    item_dummies = pd.get_dummies(data['top_item'], prefix='item').astype(int)
    data = pd.concat([data, item_dummies], axis=1)
    
    # Drop unnecessary columns
    data.drop(columns=['Time', 'Category', 'Item', 'top_item'], inplace=True)
    
    return data

# Clean the data
cleaned_data = clean_data(data)

# Perform EDA
eda_results = perform_eda(cleaned_data)

# Define top categories and top items
top_categories = eda_results['category_demand'].head(4).index.tolist()
top_items = eda_results['item_demand'].head(10).index.tolist()

# Feature engineering
engineered_data = feature_engineering(cleaned_data, top_categories, top_items)

# Print results
print(engineered_data.head())


# Step 1: Print column names
print("Column Names:\n", engineered_data.columns.tolist())

# Step 2: Check data types
print("\nData Types:\n", engineered_data.dtypes)

# Step 3: Check for missing values
print("\nMissing Values:\n", engineered_data.isnull().sum())

# Step 4: Preview the first 5 rows of the data
print("\nPreview of Engineered Data:\n", engineered_data.head())

# Step 5: Describe the numerical features
print("\nStatistical Summary of Numerical Features:\n", engineered_data.describe())


# Remove rows with negative or zero Qty (optional based on business rules)
engineered_data = engineered_data[engineered_data['Qty'] > 0]

engineered_data['rolling_avg_7day'] = engineered_data.groupby('item_takeaway')['Qty'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

print(engineered_data['Qty'].describe())
print(engineered_data['rolling_avg_7day'].describe())

engineered_data['Qty'].value_counts().plot(kind='bar')


# Let's split the data into training and testing sets
from sklearn.model_selection import train_test_split

# Define the features and target variable
X = engineered_data.drop(columns=['Qty', 'Date'])
y = engineered_data['Qty']

# Import the necessary libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor



# Map `day_of_week` to integers (Monday=0, Sunday=6)
day_of_week_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
X['day_of_week'] = X['day_of_week'].map(day_of_week_mapping)

# Verify encoding
print(f"Unique values in 'day_of_week' after encoding: {X['day_of_week'].unique()}\n")

# Step 2: Perform K-Fold Cross-Validation
print("Starting K-Fold Cross-Validation with XGBoost...\n")


# Define the XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective="reg:squarederror"
)

# Define K-Fold Cross-Validation (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metric (RMSE)
scoring = make_scorer(mean_squared_error, squared=False)

# Perform Cross-Validation
cv_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring=scoring)

# Print Cross-Validation Results
print(f"Unweighted Model Cross-Validation RMSE Scores: {cv_scores}")

print(f"Cross-Validation RMSE Scores for Each Fold: {cv_scores}")
print(f"Average RMSE Across Folds: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of RMSE: {np.std(cv_scores):.4f}\n")

# Step 3: Split Data into Training and Testing Sets
print("Splitting Data into Training and Testing Sets...\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset shapes for confirmation
print(f"Training Features Shape: {X_train.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Testing Target Shape: {y_test.shape}\n")

# Step 4: Confirm Workflow is Complete
print("Cross-Validation and Data Splitting Complete! Dataset is ready for training.")



from sklearn.metrics import mean_squared_error

# Train the XGBoost model on the training data
print("Training the XGBoost model...\n")
xgb_model.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...\n")
y_pred = xgb_model.predict(X_test)

# Evaluate the model on the test set using RMSE
rmse_test = mean_squared_error(y_test, y_pred, squared=False)

# Print the results
print(f"Test Set RMSE: {rmse_test:.4f}")



# Plot actual vs. predicted demand
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.title("Actual vs. Predicted Demand")
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.legend()
plt.grid(True)
plt.show()

# Define XGBoost model with L2 regularization (Ridge)
xgb_model_l2 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    reg_lambda=1,  # L2 regularization term
    random_state=42,
    objective="reg:squarederror"
)


# Train the model
print("Training XGBoost with L2 Regularization...\n")
xgb_model_l2.fit(X_train, y_train)

# Predict on test set
print("Evaluating the L2-regularized model...\n")
y_pred_l2 = xgb_model_l2.predict(X_test)

# Calculate RMSE
rmse_l2 = mean_squared_error(y_test, y_pred_l2, squared=False)
print(f"Test Set RMSE (L2 Regularized Model): {rmse_l2:.4f}")

# Create a rolling average for actual demand
actual_smoothed = actual_demand.rolling(window=7, min_periods=1).mean()

# Generate future dates for predictions and apply rolling average
future_dates = pd.date_range(start=actual_dates.max() + pd.Timedelta(days=1), periods=60)
future_features = pd.DataFrame({
    'Date': future_dates,
    'day_of_week': future_dates.day_name().map(day_of_week_mapping),
    'hour': 12,  # Standardized hour
    'is_weekend': future_dates.day_name().isin(['Saturday', 'Sunday']).astype(int),
    'month': future_dates.month,
    'week_of_year': future_dates.isocalendar().week,
    'rolling_avg_7day': actual_smoothed.iloc[-7:].mean()  # Use last known rolling avg
})

# Add other features for prediction (set to zero for simplicity)
for col in [col for col in engineered_data.columns if col.startswith('item_') or col.startswith('is_')]:
    future_features[col] = 0

# Align columns and predict
future_features = future_features[X.columns]
future_demand = xgb_model.predict(future_features)

# Smooth the predicted demand
future_smoothed = pd.Series(future_demand).rolling(window=7, min_periods=1).mean()

# Combine actual and predicted dates for plotting
all_dates = pd.concat([actual_dates, pd.Series(future_dates, name='Date')])
all_demand = pd.concat([actual_smoothed, future_smoothed])

# Plot smoothed demand
plt.figure(figsize=(14, 8))
plt.plot(actual_dates, actual_smoothed, label="Actual Demand (Smoothed)", color="blue", linewidth=2)
plt.plot(future_dates, future_smoothed, label="Predicted Demand (Smoothed)", color="orange", linestyle="--", linewidth=2)
plt.axvline(x=future_dates.min(), color="red", linestyle="--", label="Prediction Start", linewidth=1)
plt.title("Smoothed Daily Demand (Actual and Predicted for Next 2 Months)")
plt.xlabel("Date")
plt.ylabel("Demand (Qty)")
plt.legend()
plt.grid(True)
plt.show()
