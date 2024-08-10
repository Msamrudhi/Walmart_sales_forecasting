import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess data
@st.cache_data
def load_data():
    st.write("Loading data...")
    data = pd.read_csv('Walmart.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['YearMonth'] = data['Date'].dt.to_period('M')
    
    monthly_data = data.groupby('YearMonth').agg({
        'Weekly_Sales': 'sum',
        'Holiday_Flag': 'mean',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'CPI': 'mean',
        'Unemployment': 'mean'
    }).reset_index()
    
    # Feature Engineering: Create lagged features and moving averages
    monthly_data['Lag1_Sales'] = monthly_data['Weekly_Sales'].shift(1)
    monthly_data['Lag2_Sales'] = monthly_data['Weekly_Sales'].shift(2)
    monthly_data['MA3_Sales'] = monthly_data['Weekly_Sales'].rolling(window=3).mean()
    monthly_data.dropna(inplace=True)  # Drop missing values after lagging
    
    st.write("Data loaded successfully.")
    return monthly_data

# Function to train Random Forest model with hyperparameter tuning and cross-validation
def train_random_forest(data):
    try:
        X = data[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Lag1_Sales', 'Lag2_Sales', 'MA3_Sales']]
        y = data['Weekly_Sales']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.write("Data split and scaled successfully.")
        
        # Initialize the Random Forest model
        rf_model = RandomForestRegressor(random_state=42)
        
        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_absolute_error')
        
        st.write("Starting grid search...")
        grid_search.fit(X_train_scaled, y_train)
        st.write("Grid search completed.")
        
        # Best model after hyperparameter tuning
        best_rf_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_rf_model.predict(X_test_scaled)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write("Model training completed.")
        return X_test.index, y_test, y_pred, mae, r2, grid_search.best_params_

    except Exception as e:
        st.write(f"Error during model training: {e}")

# Function to plot predictions
def plot_predictions(dates, y_test, y_pred):
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(dates, y_test, color='blue', marker='o', label='Actual Sales')
        plt.plot(dates, y_pred, color='red', linestyle='--', marker='o', label='Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Actual vs Predicted Sales')
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.write(f"Error during plotting: {e}")

# Streamlit app
def main():
    st.title('Product Demand Prediction Dashboard')
    
    st.write("""
    This dashboard uses a Random Forest model with hyperparameter tuning and feature engineering to predict the demand (monthly sales) of a product based on various factors.
    """)
    
    data_load_state = st.text('Loading data...')
    monthly_data = load_data()
    data_load_state.text('Loading data...done!')
    
    st.subheader('Monthly Aggregated Data')
    st.write(monthly_data.head())
    
    if st.button('Train Optimized Random Forest Model'):
        dates, y_test, y_pred, mae, r2, best_params = train_random_forest(monthly_data)
        
        if y_test is not None and y_pred is not None:
            st.subheader('Model Performance')
            st.write(f'Mean Absolute Error: {mae}')
            st.write(f'R-Squared: {r2}')
            #st.write(f'Best Parameters: {best_params}')
            
            st.subheader('Actual vs Predicted Sales')
            plot_predictions(dates, y_test, y_pred)
        else:
            st.write("Model training failed. Please check the logs above.")

if __name__ == '__main__':
    main()
