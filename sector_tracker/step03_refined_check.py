import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/gdp_merged_data.csv"
TARGET_COL = "GDP_Growth"

# We remove broad "Leisure" categories to force focus on "Services"
DROP_COLS = [
    "Hobbies___Leisure", 
    "Arts___Entertainment", 
    "Games", 
    "Entertainment_Media", 
    "Entertainment_Industry", 
    "Books___Literature",
    "Gifts___Special_Event_Items"
]

def run_refined_diagnostics():
    print("--- REFINED DIAGNOSTIC START ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # Identify Target vs Features
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # 2. DROP THE NOISY COLUMNS
    # Check if they exist before dropping to avoid errors
    existing_drops = [c for c in DROP_COLS if c in X.columns]
    X_refined = X.drop(columns=existing_drops)
    
    print(f"Original Categories: {X.shape[1]}")
    print(f"Dropped: {len(existing_drops)} categories ({existing_drops})")
    print(f"Remaining Categories: {X_refined.shape[1]}")
    print("-" * 30)

    # 3. PCA Analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_refined)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    print(f"PC1 Variance: {var_ratio[0]:.2%} (Cumulative: {cum_var[0]:.2%})")
    print(f"PC2 Variance: {var_ratio[1]:.2%} (Cumulative: {cum_var[1]:.2%})")

    # 4. Lag Analysis (Refined PC2 vs GDP)
    pc2 = pca.transform(X_scaled)[:, 1]
    
    print("\n--- LAG ANALYSIS (Refined PC2 vs GDP) ---")
    correlations = []
    lags = [0, 1, 2, 3]
    
    for lag in lags:
        temp_df = pd.DataFrame({'PC2': pc2, 'GDP': y.values}, index=df.index)
        temp_df['PC2_Shifted'] = temp_df['PC2'].shift(lag)
        valid_data = temp_df.dropna()
        
        corr = valid_data['PC2_Shifted'].corr(valid_data['GDP'])
        correlations.append(corr)
        print(f"Lag {lag}: Correlation = {corr:.4f}")
        
    best_lag = np.argmax(np.abs(correlations))
    best_corr = correlations[best_lag]
    print(f"\n>> RESULT: Best Lag is {best_lag} with Correlation {best_corr:.4f}")
    
    if abs(best_corr) > 0.35:
        print(">> STATUS: GOOD. Correlation is strong enough for modeling.")
    else:
        print(">> STATUS: WEAK. We may need to use Ridge Regression.")

if __name__ == "__main__":
    run_refined_diagnostics()