import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/gdp_merged_data.csv"
TARGET_COL = "GDP_Growth"

def run_diagnostics():
    print("--- DIAGNOSTIC START ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # Identify Target vs Features
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        # Fallback if column name differs
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        print(f"Note: '{TARGET_COL}' not found. Using last column '{y.name}' as target.")

    print(f"Dataset: {X.shape[0]} Quarters, {X.shape[1]} Categories")

    # 2. PCA Analysis (The Scree Plot)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    print("\n--- PCA VARIANCE EXPLAINED ---")
    print(f"PC1: {var_ratio[0]:.2%} (Cumulative: {cum_var[0]:.2%})")
    print(f"PC2: {var_ratio[1]:.2%} (Cumulative: {cum_var[1]:.2%})")
    print(f"PC3: {var_ratio[2]:.2%} (Cumulative: {cum_var[2]:.2%})")
    print(f"PC4: {var_ratio[3]:.2%} (Cumulative: {cum_var[3]:.2%})")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_var)+1), cum_var, marker='o', linestyle='--')
    plt.axhline(0.80, color='red', linestyle=':', label='80% Threshold')
    plt.title("PCA Scree Plot: Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.grid(True)
    plt.legend()
    plt.savefig("data_weekly/pca_scree_plot.png")
    print("Graph saved: data_weekly/step1_scree_plot.png")

    # 3. Lag Analysis (Cross-Correlation)
    # We check the correlation of the dominant signal (PC1) vs GDP
    pc1 = pca.transform(X_scaled)[:, 0] # Extract just PC1
    
    print("\n--- LAG ANALYSIS (PC1 vs GDP) ---")
    print("Does Search Data predict GDP instantly (Lag 0) or early (Lag 1+)?")
    
    correlations = []
    lags = [0, 1, 2, 3] # Quarters
    
    for lag in lags:
        # Create a temporary dataframe to align dates
        temp_df = pd.DataFrame({'PC1': pc1, 'GDP': y.values}, index=df.index)
        
        # Shift PC1 forward (e.g., PC1 from Q1 aligns with GDP from Q2)
        temp_df['PC1_Shifted'] = temp_df['PC1'].shift(lag)
        
        # Drop NaNs created by shifting
        valid_data = temp_df.dropna()
        
        # Calculate Correlation
        corr = valid_data['PC1_Shifted'].corr(valid_data['GDP'])
        correlations.append(corr)
        print(f"Lag {lag} (Search leads by {lag*3} months): Correlation = {corr:.4f}")
        
    best_lag = np.argmax(np.abs(correlations))
    print(f"\n>> RECOMMENDATION: Use Lag {best_lag}")

if __name__ == "__main__":
    run_diagnostics()