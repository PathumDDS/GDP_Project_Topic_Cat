import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/weekly_preprocessed_data.csv"
OUTPUT_FILE = "data_weekly/weekly_pca_scores.csv"
PLOT_FILE = "data_weekly/weekly_pca_eigenvalues.png"

def perform_dynamic_pca():
    print("--- STEP 1: Loading Weekly Preprocessed Data ---")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File {INPUT_FILE} not found. Run preprocessing first.")
        return
        
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    num_variables = df.shape[1]
    print(f"Loaded {num_variables} variables across {df.shape[0]} weeks.")

    print("\n--- STEP 2: Standardization ---")
    # Standardization is mandatory so the covariance matrix becomes a correlation matrix,
    # making the explained_variance_ exactly equal to the eigenvalues.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    print("\n--- STEP 3: The Kaiser-Guttman Evaluation ---")
    # Fit PCA on the entire dataset to calculate all eigenvalues
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    eigenvalues = pca_full.explained_variance_
    variance_ratios = pca_full.explained_variance_ratio_
    
    # Dynamically calculate how many components have an eigenvalue > 1.0
    valid_components = sum(eigenvalues > 1.0)
    
    print(f"Mathematical Threshold: Eigenvalue (λ) > 1.0")
    print(f"Dynamically Retained Components: {valid_components}\n")
    
    # Print the breakdown to see exactly where the cutoff happens
    print(f"{'Component':<10} | {'Eigenvalue (λ)':<15} | {'Variance':<10} | {'Status'}")
    print("-" * 60)
    
    # Show the retained components plus a few dropped ones for context
    display_limit = min(valid_components + 3, len(eigenvalues))
    for i in range(display_limit):
        eigen_val = eigenvalues[i]
        var_pct = variance_ratios[i] * 100
        status = "[RETAINED]" if eigen_val > 1.0 else "[DROPPED]"
        print(f"PC{i+1:<8} | λ = {eigen_val:<11.4f} | {var_pct:>5.2f}%    | {status}")
        
    total_retained_var = np.sum(variance_ratios[:valid_components]) * 100
    print("-" * 60)
    print(f"Cumulative Variance of Retained PCs: {total_retained_var:.2f}%")
    print("-" * 60)

    # Generate the visual justification plot
    plt.figure(figsize=(10, 6))
    max_plot = min(max(valid_components + 5, 10), len(eigenvalues))
    
    plt.plot(range(1, max_plot + 1), eigenvalues[:max_plot], marker='o', linestyle='-', color='b', label='Eigenvalue (λ)')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Kaiser Threshold (λ = 1.0)')
    plt.axvline(x=valid_components, color='g', linestyle=':', label=f'Cutoff: {valid_components} PCs')
    
    plt.title('Scree Plot: Kaiser-Guttman Criterion for Feature Retention')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue (λ)')
    plt.xticks(range(1, max_plot + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f"-> Diagnostic plot saved to: {PLOT_FILE}")

    print(f"\n--- STEP 4: Extraction and Saving ---")
    # Transform the dataset keeping ONLY the dynamically justified components
    pca_final = PCA(n_components=valid_components)
    pca_scores = pca_final.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(
        pca_scores, 
        index=df.index, 
        columns=[f"PC{i+1}" for i in range(valid_components)]
    )
    
    df_pca.to_csv(OUTPUT_FILE)
    print(f"-> Successfully saved {valid_components} dynamically justified components to: {OUTPUT_FILE}")
    print("\nNext step: Merge this file with the quarterly GDP data using the alignment script.")

if __name__ == "__main__":
    perform_dynamic_pca()