import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
WEEKLY_PCA_FILE = "data_weekly/weekly_pca_scores.csv"
GDP_DATA_FILE = "data_weekly/gdp_sri_lanka.csv"
OUTPUT_FILE = "data_weekly/midas_ready_dataset.csv"

def prepare_midas_dataset():
    print("--- STEP 1: Loading Datasets ---")
    
    # 1. Load Weekly PCA Data (Predictors)
    df_weekly = pd.read_csv(WEEKLY_PCA_FILE, index_col=0, parse_dates=True)
    print(f"Loaded Weekly PCA: {df_weekly.shape[0]} weeks.")
    
    # 2. Load Quarterly GDP Data (Target)
    df_gdp = pd.read_csv(GDP_DATA_FILE, index_col='Date', parse_dates=True)
    print(f"Loaded Quarterly GDP: {df_gdp.shape[0]} quarters.")

    print("\n--- STEP 2: Time Alignment ---")
    # Tag every single date with its official economic quarter (e.g., '2016Q1')
    df_weekly['Quarter_Tag'] = df_weekly.index.to_period('Q')
    df_gdp['Quarter_Tag'] = df_gdp.index.to_period('Q')
    
    print("\n--- STEP 3: The Staggered Merge ---")
    # Identify the exact date of the LAST week for each quarter in the weekly data
    last_weeks = df_weekly.reset_index().groupby('Quarter_Tag')['index'].max()
    
    # Initialize the GDP column with empty values (NaNs) for all 510 weeks
    df_weekly['GDP_Growth'] = np.nan
    
    # Map the GDP values ONLY to those specific last-week dates
    for quarter, last_date in last_weeks.items():
        if quarter in df_gdp['Quarter_Tag'].values:
            gdp_value = df_gdp.loc[df_gdp['Quarter_Tag'] == quarter, 'GDP_Growth'].iloc[0]
            df_weekly.loc[last_date, 'GDP_Growth'] = gdp_value
            
    # Clean up the temporary merge tags
    df_weekly = df_weekly.drop(columns=['Quarter_Tag'])
    
    print(f"\n--- STEP 4: Verification ---")
    print(f"Total Weeks: {len(df_weekly)}")
    # This should roughly equal the number of quarters in your GDP file
    print(f"Total GDP Data Points Placed: {df_weekly['GDP_Growth'].notna().sum()}")
    
    # Save the final MIDAS-ready dataset
    df_weekly.to_csv(OUTPUT_FILE)
    print(f"\n-> SUCCESS: MIDAS dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_midas_dataset()