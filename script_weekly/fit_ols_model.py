import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # <--- CHANGED THIS
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/pca_dataset_final.csv"
TARGET_COL = "GDP_Growth"

def fit_bold_model():
    print("--- FITTING BOLD MODEL (Standard OLS) ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    X = df[['PC1', 'PC2', 'PC3']]
    y = df[TARGET_COL]

    # 2. Train (Leave-One-Out)
    loo = LeaveOneOut()
    preds = []
    actuals = []
    dates = []
    
    # We use Standard Linear Regression
    # It allows the model to predict extreme values (like -10%)
    model = LinearRegression()
    
    print("Training...", end="")
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        
        preds.append(pred)
        actuals.append(y_test.values[0])
        dates.append(y_test.index[0])
        print(".", end="")
        
    print(" Done!")

    # 3. Results
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    
    print(f"\n--- BOLD MODEL RESULTS ---")
    print(f"RMSE Error: {rmse:.4f}")
    print(f"R2 Score:   {r2:.4f}")
    
    # 4. Plot
    results = pd.DataFrame({'Actual': actuals, 'Predicted': preds}, index=dates).sort_index()

    results.to_csv("data_weekly/linear_reg_output.csv")
    print("\n-> Final predictions saved to: data_weekly/linear_reg_output.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Actual'], label='Official GDP', color='black', linewidth=2)
    # Plotting the prediction in Blue to distinguish it
    plt.plot(results.index, results['Predicted'], label='Bold Prediction', color='blue', linestyle='--', linewidth=2)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f"Sri Lanka GDP Tracker (Bold OLS) | RMSE: {rmse:.2f}")
    plt.legend()
    plt.savefig("data_weekly/linear_reg_plot.png")
    print("\n-> Graph saved to: data_weekly/linear_reg_plot.png")

if __name__ == "__main__":
    fit_bold_model()