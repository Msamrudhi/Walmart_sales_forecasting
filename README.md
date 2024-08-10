project Name - walmart sales forecasting.
project for walmart sparkthon.

# Product Demand Prediction Dashboard

## Overview

This project is a dashboard built with Streamlit that uses a Random Forest model to predict product demand based on various features. The model incorporates hyperparameter tuning and feature engineering to provide accurate monthly sales predictions.

## Features

- **Data Loading and Preprocessing**: Loads and preprocesses Walmart sales data, aggregating it monthly and creating additional features like lagged sales and moving averages.
- **Model Training**: Utilizes a Random Forest Regressor with hyperparameter tuning through GridSearchCV and cross-validation using TimeSeriesSplit.
- **Performance Metrics**: Displays model performance metrics such as Mean Absolute Error (MAE) and R-Squared (R²).
- **Visualization**: Plots actual vs. predicted sales to visualize the model's performance.

## Requirements

To run this application, you'll need to install the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the required packages using pip:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
