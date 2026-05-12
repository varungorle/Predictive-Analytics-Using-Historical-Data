import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# -----------------------------------
# LOAD DATASET
# -----------------------------------

print("Loading dataset...")

df = pd.read_csv(
    "train.csv.zip",
    encoding='latin1'
)

print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------------
# DATASET INFORMATION
# -----------------------------------

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------------
# DATA CLEANING
# -----------------------------------

print("\nCleaning data...")

# Remove missing values
df = df.dropna()

# Convert Order Date column to datetime
# Dataset uses DD/MM/YYYY format
df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    dayfirst=True
)

# Sort data by date
df = df.sort_values('Order Date')

print("\nData cleaned successfully!")

# -----------------------------------
# CREATE MONTHLY SALES DATA
# -----------------------------------

print("\nCreating monthly sales data...")

monthly_sales = df.groupby(
    pd.Grouper(
        key='Order Date',
        freq='ME'
    )
)['Sales'].sum().reset_index()

print("\nMonthly Sales Data:")
print(monthly_sales.head())

# -----------------------------------
# VISUALIZE HISTORICAL SALES TREND
# -----------------------------------

print("\nDisplaying sales trend graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales['Order Date'],
    monthly_sales['Sales']
)

plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.grid(True)

plt.show()

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

print("\nSplitting training and testing data...")

train = monthly_sales[:-12]
test = monthly_sales[-12:]

print("Training Data Size:", len(train))
print("Testing Data Size:", len(test))

# -----------------------------------
# BUILD ARIMA MODEL
# -----------------------------------

print("\nTraining ARIMA model...")

model = ARIMA(
    train['Sales'],
    order=(5, 1, 0)
)

model_fit = model.fit()

print("\nModel trained successfully!")

print("\nModel Summary:")
print(model_fit.summary())

# -----------------------------------
# MAKE PREDICTIONS
# -----------------------------------

print("\nForecasting future sales...")

forecast = model_fit.forecast(
    steps=12
)

print("\nPredicted Sales:")
print(forecast)

# -----------------------------------
# EVALUATE MODEL
# -----------------------------------

print("\nEvaluating model performance...")

mae = mean_absolute_error(
    test['Sales'],
    forecast
)

print("\nMean Absolute Error (MAE):", mae)

# -----------------------------------
# VISUALIZE PREDICTIONS
# -----------------------------------

print("\nDisplaying prediction graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    test['Order Date'],
    test['Sales'],
    label='Actual Sales'
)

plt.plot(
    test['Order Date'],
    forecast,
    label='Predicted Sales'
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

# -----------------------------------
# SAVE PREDICTIONS
# -----------------------------------

print("\nSaving prediction results...")

# Create outputs folder automatically
os.makedirs(
    "outputs",
    exist_ok=True
)

# Create results dataframe
results = pd.DataFrame({
    'Date': test['Order Date'].values,
    'Actual Sales': test['Sales'].values,
    'Predicted Sales': forecast.values
})

# Save CSV file
results.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")

# -----------------------------------
# FINAL MESSAGE
# -----------------------------------

print("\nProject Execution Completed Successfully!")