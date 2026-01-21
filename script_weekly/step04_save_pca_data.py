import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/final_dataset_ready.csv"
OUTPUT_FILE = "data_weekly/pca_dataset_final.csv"
TARGET_COL = "GDP_Growth"
N_COMPONENTS = 3  # Keeping top 3 is standard safety, even if #1 is dominant

def save_pca_data():
    print("--- SAVING COMPRESSED DATASET (PCA) ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # 2. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Transform (Squash 160 -> 3)
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Create New Dataframe
    cols = [f"PC{i+1}" for i in range(N_COMPONENTS)]
    df_pca = pd.DataFrame(X_pca, columns=cols, index=df.index)
    df_pca[TARGET_COL] = y  # Add GDP back
    
    # 5. Save
    df_pca.to_csv(OUTPUT_FILE)
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Shape: {df_pca.shape} (3 Columns instead of 160!)")
    
    # 6. Visual Proof
    # Does PC1 actually look like GDP?
    plt.figure(figsize=(10, 5))
    
    # We plot PC1 and GDP on different axes just to compare the SHAPE
    ax1 = plt.gca()
    ax1.plot(df_pca.index, df_pca['PC1'], color='blue', label='Principal Component 1 (Search Data)')
    ax1.set_ylabel("Search Trend Intensity", color='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(df_pca.index, df_pca[TARGET_COL], color='black', linestyle='--', label='Actual GDP')
    ax2.set_ylabel("GDP Growth (%)", color='black')
    
    plt.title("Visual Check: Does Search Data (PC1) match GDP?")
    plt.savefig("data_weekly/pca_vs_gdp_check.png")
    print("Graph saved: data_weekly/pca_vs_gdp_check.png")

if __name__ == "__main__":
    save_pca_data()