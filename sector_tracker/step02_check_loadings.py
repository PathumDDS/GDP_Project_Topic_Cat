import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data_weekly/gdp_merged_data.csv"
TARGET_COL = "GDP_Growth"

def check_loadings():
    # Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df.iloc[:, :-1]

    # Run PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    # Extract Loadings (The "Ingredients" of PC1)
    # This tells us which category contributes most to the signal
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1_Impact', 'PC2_Impact'], 
        index=X.columns
    )
    
    # Sort by PC1 influence
    print("\n--- WHAT IS INSIDE PC1? (Top Contributors) ---")
    print(loadings['PC1_Impact'].abs().sort_values(ascending=False))
    
    # Save for review
    loadings.to_csv("data_weekly/pca_loadings.csv")
    print("\nFull loadings saved to data_weekly/pca_loadings.csv")

if __name__ == "__main__":
    check_loadings()