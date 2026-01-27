import pandas as pd
import os

# --- CONFIGURATION ---
TRENDS_FILE = "data_weekly/master_data.csv"   # Already Quarterly
GDP_FILE = "data_weekly/gdp_sri_lanka.csv"    # Target Variable
OUTPUT_FILE = "data_weekly/gdp_merged_data.csv"

def load_and_merge():
    print("--- STEP 1: Loading Data ---")
    
    # 1. Load Trends Data
    if not os.path.exists(TRENDS_FILE):
        print(f"ERROR: Could not find {TRENDS_FILE}")
        return None
    
    # We use index_col=0 to ensure the Date column is used as the Index
    df_trends = pd.read_csv(TRENDS_FILE, index_col=0, parse_dates=True)
    print(f"Trends Data Loaded: {df_trends.shape} (Rows, Cols)")

    # 2. Load GDP Data
    if not os.path.exists(GDP_FILE):
        print(f"ERROR: Could not find {GDP_FILE}")
        return None
        
    df_gdp = pd.read_csv(GDP_FILE, index_col=0, parse_dates=True)
    print(f"GDP Data Loaded:    {df_gdp.shape} (Rows, Cols)")

    # 3. Merge (Inner Join)
    print("\n--- STEP 2: Merging Datasets ---")
    
    # This matches the dates. If a date (e.g., 2015-03-31) exists in BOTH files, it is kept.
    master_df = pd.merge(df_trends, df_gdp, left_index=True, right_index=True, how='inner')
    
    print(f"Merge Complete.")
    print(f"Final Dataset Shape: {master_df.shape}")
    
    # 4. Data Quality Check
    if master_df.empty:
        print("\n[CRITICAL WARNING] The merged dataset is EMPTY (0 rows).")
        print("Check your CSV dates. They must match exactly (e.g., 2016-03-31 vs 2016-03-31).")
        print(f"Trends Date Format Example: {df_trends.index[0]}")
        print(f"GDP Date Format Example:    {df_gdp.index[0]}")
    else:
        # 5. Save
        master_df.to_csv(OUTPUT_FILE)
        print(f"\n--- STEP 3: Saved to {OUTPUT_FILE} ---")
        print("You can now proceed to Model Selection.")
    
    return master_df

if __name__ == "__main__":
    load_and_merge()