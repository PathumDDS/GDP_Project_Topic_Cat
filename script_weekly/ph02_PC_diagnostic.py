import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
WEEKLY_DATA_FILE = "data_weekly/weekly_preprocessed_data.csv"
GDP_DATA_FILE = "data_weekly/gdp_sri_lanka.csv" # Ensure this has a date index and GDP column
MAX_PCS_TO_TEST = 10

def justify_pc_selection():
    print("--- STEP 1: Extracting Top 10 PCs ---")
    df_weekly = pd.read_csv(WEEKLY_DATA_FILE, index_col=0, parse_dates=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_weekly)
    
    # Extract 10 components to test
    pca = PCA(n_components=MAX_PCS_TO_TEST)
    pca_scores = pca.fit_transform(X_scaled)
    
    # Print the variance for the first 10 to see the drop-off
    print("\nExplained Variance for Top 10 PCs:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var*100:.2f}%")
        
    df_pca_weekly = pd.DataFrame(
        pca_scores, 
        index=df_weekly.index, 
        columns=[f"PC{i+1}" for i in range(MAX_PCS_TO_TEST)]
    )

    print("\n--- STEP 2: Aggregating & Merging for OLS Diagnostic ---")
    # Convert weekly PCs to Quarterly averages JUST for the OLS test
    df_pca_quarterly = df_pca_weekly.resample('Q').mean()
    df_pca_quarterly.index = df_pca_quarterly.index.to_period('Q')
    
    df_gdp = pd.read_csv(GDP_DATA_FILE, index_col=0, parse_dates=True)
    df_gdp.index = df_gdp.index.to_period('Q')
    gdp_col = df_gdp.columns[0] # Assuming GDP is the first column
    
    # Merge datasets on the quarter index
    merged_df = df_pca_quarterly.join(df_gdp[[gdp_col]], how='inner').dropna()
    
    # Apply the 1-Quarter Lead (Shift GDP backwards by 1 to test predictive power)
    merged_df['Target_GDP_Lead1'] = merged_df[gdp_col].shift(-1)
    merged_df = merged_df.dropna()
    
    y = merged_df['Target_GDP_Lead1']
    
    print(f"\n--- STEP 3: OLS Sequential Testing (n={len(merged_df)} quarters) ---")
    print(f"{'Model':<15} | {'Adj R-Squared':<15} | {'BIC':<10} | {'Newest PC p-value'}")
    print("-" * 65)
    
    best_bic = float('inf')
    optimal_pcs = 0
    
    for i in range(1, MAX_PCS_TO_TEST + 1):
        # Select the first 'i' principal components
        features = [f"PC{j}" for j in range(1, i + 1)]
        X = merged_df[features]
        X = sm.add_constant(X)
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Get metrics
        adj_r2 = model.rsquared_adj
        bic = model.bic
        # p-value of the most recently added component (last one in the summary)
        p_val_newest = model.pvalues.iloc[-1] 
        
        print(f"PCs 1 to {i:<2}   | {adj_r2:<15.4f} | {bic:<10.2f} | {p_val_newest:.4f}")
        
        # Track the model with the lowest BIC
        if bic < best_bic:
            best_bic = bic
            optimal_pcs = i

    print("\n--- CONCLUSION ---")
    print(f"According to BIC, the mathematically optimal number of components to retain is: {optimal_pcs}")
    print("This provides strict supervised justification for the extended abstract.")

if __name__ == "__main__":
    justify_pc_selection()