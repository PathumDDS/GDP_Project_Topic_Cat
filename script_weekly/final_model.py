import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/pca_dataset_final.csv"
TARGET_COL = "GDP_Growth"
LAG = 1  # 3 Months

# --- CONFIGURATION ---
# The file used to creating the PCA originally (Source of Truth)
ORIGINAL_PCA_INPUT = "data_weekly/gdp_merged_data.csv" 
# The Raw Weekly Data you want to predict on (2015-2025)
WEEKLY_TARGET_FILE = "data_weekly/weekly_preprocessed.csv"
# The GDP file for plotting
GDP_FILE = "data_weekly/srilanka_gdp_actuals.csv"

def train_and_evaluate_clean(output_name):
    print(f"--- RUNNING CLEAN MODEL (Reproducing RMSE 2.68) ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # 2. DELETE 2020 FIRST (The Key Step)
    # This ensures we don't even use 2020 data as inputs for 2021
    df_clean = df[~df.index.year.isin([2020])]
    
    X_raw = df_clean[['PC1', 'PC2', 'PC3']]
    y_raw = df_clean[TARGET_COL]
    
    # 3. Apply Lag (Shift logic on clean data only)
    # This effectively connects 2019-Q4 to 2021-Q1
    X_lagged = X_raw.shift(LAG)
    data = pd.concat([X_lagged, y_raw], axis=1).dropna()
    
    # 4. Leave-One-Out Validation
    loo = LeaveOneOut()
    model = LinearRegression()
    
    preds = []
    actuals = []
    dates = []
    
    for train_idx, test_idx in loo.split(data):
        X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
        y_train, y_test = data.iloc[train_idx][TARGET_COL], data.iloc[test_idx][TARGET_COL]
        
        # We need to fit on X (inputs) and y (target)
        # Note: X_train includes the TARGET column in 'data', so we split it:
        X_tr = X_train[['PC1', 'PC2', 'PC3']]
        y_tr = y_train # y_train is already the Series
        X_te = X_test[['PC1', 'PC2', 'PC3']]
        
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)[0]
        
        preds.append(pred)
        actuals.append(y_test) # it's a scalar value from the series
        dates.append(data.index[test_idx[0]])

    # 5. Metrics
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    
    print(f"RMSE: {rmse:.4f} (Should be ~2.68)")
    print(f"R2:   {r2:.4f}")

    # 5. Save to disk
    joblib.dump(model, 'data_weekly/india_gdp_model.pkl')
    print("Model saved to: data_weekly/india_gdp_model.pkl")
    print(f"Coefficients: {model.coef_}")
    
    # 6. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actuals, color='black', linewidth=2, label='Actual GDP')
    plt.plot(dates, preds, color='green', linestyle='--', marker='o', markersize=4, label='Clean Model Prediction')
    
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.title(f"Clean Model (Excluding 2020) | RMSE: {rmse:.2f}")
    plt.ylabel("YoY GDP Growth (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"data_weekly/model_{output_name}.png")
    print(f"Graph saved to: data_weekly/model_{output_name}.png")


def train_and_evaluate_dirty(output_name):
    print(f"\n--- RUNNING DIRTY MODEL (With 2020) ---")
    
    # 1. Load Data (Keep Everything)
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    X_raw = df[['PC1', 'PC2', 'PC3']]
    y_raw = df[TARGET_COL]
    
    # 2. Shift First (Standard Logic)
    X_lagged = X_raw.shift(LAG)
    data = pd.concat([X_lagged, y_raw], axis=1).dropna()
    
    # 3. Leave-One-Out
    loo = LeaveOneOut()
    model = LinearRegression()
    
    preds = []
    actuals = []
    dates = []
    
    for train_idx, test_idx in loo.split(data):
        X_train, X_test = data.iloc[train_idx][['PC1', 'PC2', 'PC3']], data.iloc[test_idx][['PC1', 'PC2', 'PC3']]
        y_train, y_test = data.iloc[train_idx][TARGET_COL], data.iloc[test_idx][TARGET_COL]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        
        preds.append(pred)
        actuals.append(y_test)
        dates.append(data.index[test_idx[0]])

    # 4. Metrics
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    
    print(f"RMSE: {rmse:.4f} ")
    print(f"R2:   {r2:.4f}")
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actuals, color='black', linewidth=2, label='Actual GDP')
    plt.plot(dates, preds, color='red', linestyle='--', marker='o', markersize=4, label='Model with 2020 (Failed)')
    
    # Highlight 2020
    plt.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'), color='red', alpha=0.1)
    
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.title(f"Model Including 2020 (Anomaly) | RMSE: {rmse:.2f}")
    plt.ylabel("YoY GDP Growth (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"data_weekly/model_{output_name}.png")
    print(f"Graph saved to: data_weekly/model_{output_name}.png")



if __name__ == "__main__":
    # Run the separate analyses
    train_and_evaluate_clean("clean")
    train_and_evaluate_dirty("dirty")