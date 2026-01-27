import pandas as pd
import os
import numpy as np

import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# Update this filename to match your current combined dataset
INPUT_FILE = "data_weekly/final_dataset/weekly_merged_data.csv"  
OUTPUT_FILE_WEEKLY = "data_weekly/weekly_preprocessed.csv"
OUTPUT_FILE = "data_weekly/preprocessed_data.csv"
CUTOFF_DATE = "2025-09-30"

# Strict threshold for "Empty" (NaN) data
# If a column is missing more than 5% of its data points, we throw it away.
NAN_THRESHOLD = 0.05

def handle_empty_values(df, threshold=0.05):
    """
    1. Analyzes ONLY real empty values (NaNs), not zeros.
    2. Drops variables that are missing significant chunks of data.
    3. Fills small gaps with 0 (assuming low volume).
    """
    print(f"\n--- Step 2: Handling Empty Values (NaNs) ---")
    
    # 1. SEE: Count NaNs per column
    nan_counts = df.isna().sum()
    nan_pcts = nan_counts / len(df)
    
    # Identify Bad Variables (Too many empty spots)
    bad_vars = nan_pcts[nan_pcts > threshold]
    
    print(f"Total Variables: {len(df.columns)}")
    
    # 2. REPORT: Show us the problematic ones
    if len(bad_vars) > 0:
        print(f"\n[CRITICAL] Found {len(bad_vars)} broken variables (> {threshold*100}% Empty):")
        print(bad_vars.sort_values(ascending=False).head(10))
        print("-> ACTION: Dropping these variables.")
        
        # 3. SOLVE PART A: Drop the broken ones
        df = df.drop(columns=bad_vars.index)
    else:
        print("-> No variables exceeded the failure threshold.")

    # 4. SOLVE PART B: Handle the survivors
    # If a variable has only a few missing spots (e.g. 1%), we assume 
    # the missing data implies "Low Volume" -> 0.
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"\n-> ACTION: Filling {remaining_nans} remaining empty spots with 0.")
        df = df.fillna(0)
    
    print(f"Variables Remaining: {len(df.columns)}")
    return df

def apply_log_transformation(df):
    """
    PHASE 2: Log Transformation
    Formula: y = ln(x + 1)
    
    Purpose: Stabilizes variance so huge spikes don't drown out 
    small economic signals.
    """
    print(f"\n--- Phase 2: Log Transformation (y = ln(x + 1)) ---")
    
    # 1. Apply the formula
    # np.log1p(x) is a specific numpy function that calculates ln(x + 1)
    # It is more accurate for small numbers than np.log(x + 1)
    df_log = np.log1p(df)
    
    # 2. Verification Report
    print("Transformation applied.")
    print(f"Old Max Value (approx): {df.max().max():.2f}")
    print(f"New Max Value (approx): {df_log.max().max():.2f}")
    
    return df_log


def remove_common_trend_oecd(df_log):
    """
    PHASE 3: Anchoring / Common Trend Removal
    Based on Woloszko (2020).
    
    1. Extract the smooth trend of each variable using HP Filter.
    2. Run PCA on these trends to find the 'Common Trend' (PC1).
    3. Subtract PC1 from the original log-data to remove the bias.
    """
    print(f"\n--- Phase 3: Removing Common Trend (PCA + HP Filter) ---")
    
    # --- Step A: Extract Long-Term Trends (HP Filter) ---
    # We do this because we want to run PCA on the TRENDS, not the daily noise.
    # Lambda = 270400 is the specific rule for WEEKLY data.
    print("1. Extracting smooth trends using HP Filter (This may take a moment)...")
    trends = pd.DataFrame(index=df_log.index, columns=df_log.columns)
    
    for col in df_log.columns:
        # The HP filter splits data into "Cycle" (Short term) and "Trend" (Long term)
        # We only keep the trend [1]
        cycle, trend = sm.tsa.filters.hpfilter(df_log[col], lamb=270400)
        trends[col] = trend

    # --- Step B: PCA Extraction (Finding the Internet Growth Line) ---
    print("2. Running PCA to identify the common Internet Growth signal...")
    
    # Standardize trends so big variables don't dominate small ones
    scaler = StandardScaler()
    trends_scaled = scaler.fit_transform(trends)
    
    # Extract the First Principal Component (PC1)
    pca = PCA(n_components=1)
    pca.fit(trends_scaled)
    common_factor = pca.transform(trends_scaled) # This is the raw shape of internet growth
    
    # --- Step C: Rescaling & Sign Correction ---
    # We need to make sure the factor is pointing the right way (Internet grows UP)
    # If the correlation is negative, we flip the sign.
    if np.corrcoef(common_factor.T, trends.mean(axis=1))[0,1] < 0:
        common_factor = -common_factor

    # We resize this factor to match the actual scale of your data
    # (Matches the mean and std dev of the average log-SVI)
    avg_trend = df_log.mean(axis=1)
    target_std = avg_trend.std()
    target_mean = avg_trend.mean()
    
    # Final 'Bias Line' calculation
    common_trend_final = (common_factor.flatten() * target_std) + target_mean
    
    # --- Step D: Subtraction (The Cancellation) ---
    # Formula: Cleaned_Data = Log_Data - Common_Trend
    print("3. Subtracting the common trend from all variables...")
    df_detrended = df_log.sub(common_trend_final, axis=0)
    
    return df_detrended


def calculate_yoy_growth(df):
    """
    PHASE 4: Year-on-Year Differentiation
    Formula: y_t - y_{t-52}
    
    Purpose: 
    1. Removes Seasonality (Holidays, recurring annual events).
    2. Converts the data into 'Growth Rates', which matches how GDP is measured.
    """
    print(f"\n--- Phase 4: Calculating YoY Growth (Diff 52) ---")
    
    # Calculate the difference over 52 weeks (1 year)
    df_growth = df.diff(52)
    
    # The first 52 weeks will now be NaN (Empty) because they have no previous year.
    # We must drop them.
    initial_rows = len(df_growth)
    df_growth = df_growth.dropna()
    dropped_rows = initial_rows - len(df_growth)
    
    print(f"Applied YoY differencing.")
    print(f"Dropped {dropped_rows} weeks (The first year of data).")
    print(f"Remaining Data Points: {len(df_growth)} weeks.")
    
    return df_growth

def convert_to_quarterly(df):
    """
    PHASE 5: Frequency Conversion
    Aggregation: Weekly -> Quarterly
    
    Method: Mean (Average)
    We take the average of all weeks in a quarter to represent that quarter.
    """
    print(f"\n--- Phase 5: Frequency Conversion (Weekly -> Quarterly) ---")
    
    # Resample to Quarterly frequency using the Mean
    # 'Q' stands for Quarter end.
    df_quarterly = df.resample('Q').mean()
    
    print(f"Original Weekly Rows: {len(df)}")
    print(f"New Quarterly Rows:   {len(df_quarterly)}")
    
    # Quick sanity check: 10 years should be approx 40 quarters
    return df_quarterly

def main():
    # 1. Load the Dataset
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Could not find {INPUT_FILE}")
        return

    print(f"--- Loading Data from {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # 2. Ensure the Index is actually a Date (Crucial)
    # This fixes any issues where dates are read as strings
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"Original Data Range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Check if we actually have extra data
    extra_data = df.loc[df.index > CUTOFF_DATE]
    if not extra_data.empty:
        print(f"Found {len(extra_data)} rows of data after {CUTOFF_DATE}. Removing them...")
    else:
        print(f"No data found after {CUTOFF_DATE}. The file is already short enough.")

    # 3. The "Hard Cut"
    # We slice the dataframe to keep only rows up to (and including) the cutoff
    df = df.loc[:CUTOFF_DATE]

    # 4. Verification
    print("-" * 30)
    print(f"New Data Range:      {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total Weeks:         {len(df)}")
    print(f"Total Variables:     {len(df.columns)}")
    print("-" * 30)

    # --- STEP 2: EMPTY VALUE HANDLING ---
    # We call our new function here
    df = handle_empty_values(df, threshold=NAN_THRESHOLD)

    # --- PHASE 2: LOG TRANSFORMATION ---
    # We call the new function here
    df = apply_log_transformation(df)

    # --- PHASE 3: COMMON TREND REMOVAL (NEW) ---
    # This calls the function we just added
    df = remove_common_trend_oecd(df)

    # --- PHASE 4: YoY GROWTH (NEW) ---
    df = calculate_yoy_growth(df)
    df.to_csv(OUTPUT_FILE_WEEKLY)

    # --- PHASE 5: CONVERT TO QUARTERLY (NEW) ---
    df = convert_to_quarterly(df)

    # 5. Save the Cleaned File
    df.to_csv(OUTPUT_FILE)
    print(f"Success! Aligned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()