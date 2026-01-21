import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/pca_dataset_final.csv"
TARGET_COL = "GDP_Growth"

def test_lags():
    print("--- TESTING TIME LAGS (Finding the 'Early Warning' Signal) ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # Remove Covid (2020) because we know it confuses the model
    df = df[~df.index.year.isin([2020])]
    
    # 2. Setup the Shifts
    # Lag 0 = No Shift (Current Model)
    # Lag 1 = Search predicts GDP 3 months later
    # Lag 2 = Search predicts GDP 6 months later
    lags = [0, 1, 2]
    
    best_rmse = float("inf")
    best_lag = 0
    
    plt.figure(figsize=(12, 6))
    
    # Plot Actual GDP once
    plt.plot(df.index, df[TARGET_COL], label='Actual GDP', color='black', linewidth=2, alpha=0.8)

    for lag in lags:
        print(f"\nTesting Lag {lag} (Shift {lag*3} months)...")
        
        # Shift the FEATURES down, so "Jan Search" aligns with "April GDP"
        # We use .shift(lag) on X
        X = df[['PC1', 'PC2', 'PC3']].shift(lag)
        y = df[TARGET_COL]
        
        # Drop the empty rows created by shifting
        valid_indices = ~X.isnull().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Train (Leave-One-Out)
        loo = LeaveOneOut()
        preds = []
        actuals = []
        
        model = LinearRegression()
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            preds.append(pred)
            actuals.append(y_test.values[0])

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        print(f"-> RMSE: {rmse:.4f}")
        
        # Check if winner
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag
            
        # Add to plot
        if lag > 0:
            plt.plot(y.index, preds, linestyle='--', label=f'Lag {lag} (RMSE: {rmse:.2f})')
        else:
            plt.plot(y.index, preds, linestyle=':', label=f'No Lag (RMSE: {rmse:.2f})')

    print(f"\n--- WINNER ---")
    print(f"Best Lag: {best_lag} Quarters ({best_lag*3} Months)")
    print(f"New RMSE: {best_rmse:.4f}")
    
    plt.title("Effect of Time Lags on Prediction Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("data_weekly/lag_test_results.png")
    print("Graph saved to data_weekly/lag_test_results.png")

if __name__ == "__main__":
    test_lags()