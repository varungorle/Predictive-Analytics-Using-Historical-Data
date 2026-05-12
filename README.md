# Predictive Analytics Using Historical Data

## Project Overview
This project focuses on predictive analytics using historical sales data to forecast future sales trends. The project uses the ARIMA time-series forecasting model to analyze past sales records and predict upcoming sales performance.

The main objective of this project is to understand data preprocessing, trend analysis, forecasting techniques, and predictive modeling using Python.

---

## Features
- Historical sales data analysis
- Data cleaning and preprocessing
- Monthly sales trend visualization
- Time-series forecasting using ARIMA
- Actual vs Predicted sales comparison
- Model evaluation using MAE (Mean Absolute Error)
- Prediction results exported to CSV

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

---

## Dataset Used
Superstore Sales Dataset

Dataset contains:
- Order Date
- Sales
- Product Details
- Customer Information
- Region and Category Details

---

## Project Workflow

1. Load historical sales dataset
2. Clean and preprocess the data
3. Convert date columns into datetime format
4. Aggregate monthly sales data
5. Visualize historical sales trends
6. Split dataset into training and testing data
7. Train ARIMA forecasting model
8. Predict future sales
9. Evaluate model performance
10. Save prediction results

---

## Model Used

### ARIMA (AutoRegressive Integrated Moving Average)

ARIMA is a popular time-series forecasting model used for analyzing historical trends and predicting future values based on past observations.

---

## Output

The project generates:
- Monthly sales trend graph
- Actual vs Predicted sales graph
- predictions.csv file containing forecasted sales

---

## Folder Structure

```text
Predictive Analytics Using Historical Data/
│
├── data/
│   └── sales.csv
│
├── outputs/
│   └── predictions.csv
│
├── screenshots/
│
├── main.py
├── README.md
```

---

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Step 2: Run the Project

```bash
python main.py
```

---

## Results

The ARIMA model successfully analyzed historical sales trends and generated future sales predictions. The project demonstrates the implementation of predictive analytics and forecasting techniques using Python.

---

## Learning Outcomes

Through this project, I learned:
- Data preprocessing techniques
- Time-series forecasting
- Predictive analytics concepts
- Trend analysis
- Data visualization
- Model evaluation techniques

---

## Future Improvements
- Add Streamlit dashboard
- Implement Prophet forecasting model
- Add interactive visualizations
- Deploy project online
- Compare multiple forecasting models

---

## Author

Developed as part of internship learning and predictive analytics practice project.
