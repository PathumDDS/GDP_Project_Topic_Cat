import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/pca_dataset_final.csv"
TARGET_COL = "GDP_Growth"

def fit_final_pca_model():
    print("--- FITTING FINAL MODEL (Using PCA Components) ---")
    
    # 1. Load the Cleaned PCA Data (Only 3 Inputs!)
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    X = df[['PC1', 'PC2', 'PC3']]  # Our 3 Super-Variables
    y = df[TARGET_COL]
    
    print(f"Inputs: {X.shape[1]} variables (PC1, PC2, PC3)")
    print(f"Target: GDP Growth")

    # 2. Train with Leave-One-Out Validation (Honest Testing)
    loo = LeaveOneOut()
    preds = []
    actuals = []
    dates = []
    
    # We use Ridge Regression (Linear Regression with safety)
    # It finds the best weights for PC1, PC2, PC3 automatically
    model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    
    print("\nTraining...", end="")
    
    for train_idx, test_idx in loo.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        pred = model.predict(X_test)[0]
        preds.append(pred)
        actuals.append(y_test.values[0])
        dates.append(y_test.index[0])
        print(".", end="")
        
    print(" Done!")

    # 3. Calculate Final Accuracy
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    
    print(f"\n--- FINAL REPORT CARD ---")
    print(f"RMSE Error: {rmse:.4f}")
    print(f"R2 Score:   {r2:.4f}")
    
    # 4. Save Final Predictions (This is your Product!)
    results = pd.DataFrame({
        'Actual_GDP': actuals,
        'Predicted_GDP': preds
    }, index=dates).sort_index()
    
    results.to_csv("data_weekly/ridge_output.csv")
    print("\n-> Final predictions saved to: data_weekly/final_tracker_output.csv")

    # 5. The "Money Plot"
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Actual_GDP'], label='Official GDP (Black)', color='black', linewidth=2)
    plt.plot(results.index, results['Predicted_GDP'], label='AI Tracker Prediction (Red)', color='red', linestyle='--', linewidth=2)
    
    # Add a zero line for reference
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f"Sri Lanka GDP Tracker (Final PCA Model) | RMSE: {rmse:.2f}")
    plt.ylabel("YoY Growth (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("data_weekly/ridge_plot.png")
    print("-> Graph saved to: data_weekly/ridge_plot.png")

if __name__ == "__main__":
    fit_final_pca_model()