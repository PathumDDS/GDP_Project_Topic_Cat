import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/gdp_merged_data.csv"
TARGET_COL = "GDP_Growth"

def determine_components():
    print("--- PCA ANALYSIS: Determining Optimal Components ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    X = df.drop(columns=[TARGET_COL])
    
    # 2. Standardize (Mandatory for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Run Full PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # 4. Calculate Cumulative Variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 5. Print Key Milestones
    # We want to know how many components are needed to capture 80% and 90% of the information
    n_80 = np.argmax(cumulative_variance >= 0.80) + 1
    n_90 = np.argmax(cumulative_variance >= 0.90) + 1
    
    print(f"\n--- RESULTS ---")
    print(f"Total Variables: {X.shape[1]}")
    print(f"Components needed for 80% info: {n_80}")
    print(f"Components needed for 90% info: {n_90}")
    
    # 6. Plot the "Scree Plot" (The Elbow Curve)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.80, color='r', linestyle='-', label='80% Threshold')
    plt.axhline(y=0.90, color='g', linestyle='-', label='90% Threshold')
    plt.title('PCA Explained Variance (How many components do we need?)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.grid(True)
    plt.legend()
    
    plt.savefig("data_weekly/pca_scree_plot.png")
    print(f"\nGraph saved to: data_weekly/pca_scree_plot.png")
    print("Look at the graph. Where does the curve start to flatten out?")

if __name__ == "__main__":
    determine_components()