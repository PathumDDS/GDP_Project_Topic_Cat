import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/weekly_preprocessed_data.csv"
OUTPUT_FILE = "data_weekly/weekly_pca_scores.csv"
PLOT_FILE = "data_weekly/weekly_pca_eigenvalues.png"

MODELS_DIR = "models"
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
PCA_FILE = os.path.join(MODELS_DIR, "pca_model.pkl")
HOLDOUT_YEAR = 2025 # Everything from this year onward is hidden from the math

def perform_dynamic_pca():
    print("--- STEP 1: Loading Weekly Preprocessed Data ---")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File {INPUT_FILE} not found. Run preprocessing first.")
        return
        
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    num_variables = df.shape[1]
    print(f"Loaded {num_variables} variables across {df.shape[0]} weeks.")

    # --- STRICT DATA LEAKAGE PREVENTION ---
    # Separate the training timeline from the holdout timeline
    train_mask = df.index.year < HOLDOUT_YEAR
    df_train = df[train_mask]
    
    print(f"Restricting Scaler and PCA training strictly to {df_train.index.min().date()} through {df_train.index.max().date()}.")

    print("\n--- STEP 2: Strict Out-of-Sample Standardization ---")
    scaler = StandardScaler()
    
    # FIT ONLY ON TRAINING DATA
    scaler.fit(df_train)
    
    # Transform the full dataset (2025 is transformed using 2016-2024 rules)
    X_scaled_full = scaler.transform(df) 
    X_scaled_train = scaler.transform(df_train) # Used for finding eigenvectors
    
    print("\n--- STEP 3: The Kaiser-Guttman Evaluation (Training Data Only) ---")
    pca_full = PCA()
    pca_full.fit(X_scaled_train) # FIT ONLY ON TRAINING DATA
    
    eigenvalues = pca_full.explained_variance_
    variance_ratios = pca_full.explained_variance_ratio_
    
    valid_components = sum(eigenvalues > 1.0)
    
    print(f"Mathematical Threshold: Eigenvalue (λ) > 1.0")
    print(f"Dynamically Retained Components: {valid_components}\n")
    
    # Generate the visual justification plot
    plt.figure(figsize=(10, 6))
    max_plot = min(max(valid_components + 5, 10), len(eigenvalues))
    
    plt.plot(range(1, max_plot + 1), eigenvalues[:max_plot], marker='o', linestyle='-', color='b', label='Eigenvalue (λ)')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Kaiser Threshold (λ = 1.0)')
    plt.axvline(x=valid_components, color='g', linestyle=':', label=f'Cutoff: {valid_components} PCs')
    
    plt.title('Scree Plot: Kaiser-Guttman Criterion (Strict Training Sample)')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue (λ)')
    plt.xticks(range(1, max_plot + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f"-> Diagnostic plot saved to: {PLOT_FILE}")

    print("\n--- STEP 4: Extraction and Saving ---")
    pca_final = PCA(n_components=valid_components)
    
    # FIT ONLY ON TRAINING, but TRANSFORM ALL DATA for the final CSV
    pca_final.fit(X_scaled_train)
    pca_scores_full = pca_final.transform(X_scaled_full)
    
    df_pca = pd.DataFrame(
        pca_scores_full, 
        index=df.index, 
        columns=[f"PC{i+1}" for i in range(valid_components)]
    )
    
    df_pca.to_csv(OUTPUT_FILE)
    print(f"-> Successfully saved {valid_components} leak-free components to: {OUTPUT_FILE}")

    print("\n--- STEP 5: Serializing Preprocessing Pipeline ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # We dump the strictly fitted models
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(pca_final, PCA_FILE)
    
    print(f"-> [SUCCESS] Exported pure historical Scaler rules to: {SCALER_FILE}")
    print(f"-> [SUCCESS] Exported pure historical PCA matrix to: {PCA_FILE}")

if __name__ == "__main__":
    perform_dynamic_pca()