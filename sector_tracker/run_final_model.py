import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/gdp_merged_data.csv"
OUTPUT_DIR = "data_weekly"
TARGET_COL = "GDP_Growth"

# The 7 Categories to DROP (Noise Filter)
DROP_COLS = [
    "Hobbies___Leisure", "Arts___Entertainment", "Games", 
    "Entertainment_Media", "Entertainment_Industry", 
    "Books___Literature", "Gifts___Special_Event_Items"
]

def fit_final_tracker():
    print("--- STEP 1: LOADING & FILTERING ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # 1. Separate Target & Features
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # 2. Drop Noise Categories
    existing_drops = [c for c in DROP_COLS if c in X.columns]
    X_refined = X.drop(columns=existing_drops)
    print(f"Using {X_refined.shape[1]} categories (Dropped {len(existing_drops)})")

    # 3. Lag 0 Alignment (Instant Prediction)
    # Since Lag is 0, we don't need to shift. X and y are already aligned.
    # We just drop any naturally missing rows (NaNs)
    data = pd.concat([X_refined, y], axis=1).dropna()
    X_final = data.iloc[:, :-1]
    y_final = data.iloc[:, -1]
    
    print(f"Modeling with {len(X_final)} quarters of data.")

    # --- STEP 2: CROSS-VALIDATION (LOOCV) ---
    print("\n--- STEP 2: TRAINING MODEL (Ridge Regression) ---")
    loo = LeaveOneOut()
    predictions = []
    actuals = []
    dates = []
    
    # We use Ridge because correlation is moderate (~0.36)
    model = Ridge(alpha=1.0) 
    
    for train_ix, test_ix in loo.split(X_final):
        # A. Split
        X_train, X_test = X_final.iloc[train_ix], X_final.iloc[test_ix]
        y_train, y_test = y_final.iloc[train_ix], y_final.iloc[test_ix]
        
        # B. Scale (Fit on Train)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # C. PCA (Fit on Train) - Using 2 Components
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_sc)
        X_test_pca = pca.transform(X_test_sc)
        
        # D. Fit & Predict
        model.fit(X_train_pca, y_train)
        pred = model.predict(X_test_pca)[0]
        
        predictions.append(pred)
        actuals.append(y_test.values[0])
        dates.append(X_final.index[test_ix[0]])

    # --- STEP 3: EVALUATION ---
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    print(f"\n>> FINAL RESULTS:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R2:   {r2:.4f}")

    # --- STEP 4: SAVE FINAL OUTPUT ---
    # Fit model on ALL data for future use
    scaler_final = StandardScaler()
    X_all_sc = scaler_final.fit_transform(X_final)
    
    pca_final = PCA(n_components=2)
    X_all_pca = pca_final.fit_transform(X_all_sc)
    
    model.fit(X_all_pca, y_final)
    
    # Save Results to CSV
    results_df = pd.DataFrame({
        'Actual_GDP': actuals, 
        'Tracker_Pred': predictions
    }, index=dates).sort_index()
    
    out_path = os.path.join(OUTPUT_DIR, "final_sector_tracker.csv")
    results_df.to_csv(out_path)
    print(f"\nData saved to: {out_path}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(results_df.index, results_df['Actual_GDP'], 'k-o', label='Actual GDP', alpha=0.7)
    plt.plot(results_df.index, results_df['Tracker_Pred'], 'b--', label='Tracker Prediction', linewidth=2)
    plt.axhline(0, color='red', linewidth=0.5)
    plt.title(f"Other Personal Services Tracker (Lag 0)\nRMSE: {rmse:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_tracker_plot.png"))
    print("Graph saved to: final_tracker_plot.png")

if __name__ == "__main__":
    fit_final_tracker()