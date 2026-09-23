import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODELS_DIR = "models"
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
PCA_FILE = os.path.join(MODELS_DIR, "pca_model.pkl")
WEIGHTS_FILE = os.path.join(MODELS_DIR, "midas_optimized_weights.npy")

# =====================================================================
# PART 1: THE MATHEMATICAL ENGINE (Frozen Rules)
# =====================================================================
def exponential_almon_weights(theta1, theta2, num_weeks):
    """Dynamically calculates Almon weights for the elapsed weeks in the current quarter."""
    w = np.arange(1, num_weeks + 1)
    num = np.exp(np.clip(theta1 * w + theta2 * (w**2), -50, 50))
    return num / np.sum(num)

def live_gdp_prediction(new_weekly_raw_data, previous_gdp, current_quarter_pcs=None):
    """
    Takes 1 week of raw Google Trends data, applies historical scaling/PCA, 
    and outputs the live GDP Nowcast using frozen MIDAS weights.
    """
    # 1. LOAD THE FROZEN HISTORICAL RULES
    try:
        scaler = joblib.load(SCALER_FILE)
        pca = joblib.load(PCA_FILE)
        midas_params = np.load(WEIGHTS_FILE)
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Missing model file: {e.filename}")
        print("Ensure you have run the training scripts to export the models first.")
        return None, None

    # 2. DETERMINISTIC TRANSFORMATION (159 Categories -> PCs)
    scaled_data = scaler.transform(new_weekly_raw_data)
    new_week_pcs_full = pca.transform(scaled_data) 
    
    # --- THE FIX IS HERE ---
    # Calculate exactly how many PCs the MIDAS weights were optimized for
    # Formula: Total parameters = 3 (intercept, ar, dummy) + (num_pcs * 3)
    num_optimal_pcs = (len(midas_params) - 3) // 3
    
    # Slice the PCA output to match ONLY the optimal PCs used by MIDAS
    new_week_pcs = new_week_pcs_full[:, :num_optimal_pcs]

    # 3. UPDATE THE QUARTER'S TIMELINE
    if current_quarter_pcs is not None and len(current_quarter_pcs) > 0:
        quarter_data = np.vstack([current_quarter_pcs, new_week_pcs])
    else:
        quarter_data = new_week_pcs
        
    num_weeks = quarter_data.shape[0]

    # 4. EXECUTE THE FROZEN MIDAS EQUATION
    intercept = midas_params[0]
    ar_coef = midas_params[1]
    dummy_coef = midas_params[2] 
    d_crisis = 0.0 # 0 for current/future live forecasts
    
    # Base Anchor: Intercept + AR(1) momentum from the previous quarter
    prediction = intercept + (ar_coef * previous_gdp) + (dummy_coef * d_crisis)
    
    # Add the Principal Component signals from Google Trends
    idx = 3
    for p in range(num_optimal_pcs):
        gamma = midas_params[idx]
        theta1 = midas_params[idx+1]
        theta2 = midas_params[idx+2]
        idx += 3
        
        # Calculate the Almon curve for the specific number of elapsed weeks
        weights = exponential_almon_weights(theta1, theta2, num_weeks)
        
        # Multiply the PC values by the curve and add to the final prediction
        weighted_pc = np.sum(quarter_data[:, p] * weights)
        prediction += gamma * weighted_pc
        
    return prediction, quarter_data


# =====================================================================
# PART 2: SIMULATING A REAL-WORLD CENTRAL BANK DEPLOYMENT
# =====================================================================
if __name__ == "__main__":
    
    print("=========================================================")
    print("  SRI LANKA GDP NOWCASTING: LIVE DEPLOYMENT ENGINE")
    print("=========================================================\n")
    
    official_previous_gdp = 3.2 # Mock previous quarter GDP
    
    try:
        scaler = joblib.load(SCALER_FILE)
        num_expected_categories = scaler.n_features_in_
    except:
        num_expected_categories = 159
    
    print(f"[SYSTEM] Baseline established. Previous Quarter GDP anchored at: {official_previous_gdp}%\n")
    
    # --- WEEK 1 INGESTION ---
    print("[EVENT] Ingesting newly published Google Trends data for Week 1...")
    week_1_raw_data = np.random.rand(1, num_expected_categories) * 100 
    
    current_prediction, updated_pcs = live_gdp_prediction(
        new_weekly_raw_data=week_1_raw_data, 
        previous_gdp=official_previous_gdp, 
        current_quarter_pcs=None
    )
    
    if current_prediction is not None:
        print(f">> LIVE NOWCAST (Week 1): {current_prediction:.3f}%\n")
    
    # --- WEEK 2 INGESTION ---
    print("[EVENT] Seven days elapsed. Ingesting Google Trends data for Week 2...")
    week_2_raw_data = np.random.rand(1, num_expected_categories) * 100
    
    current_prediction, updated_pcs = live_gdp_prediction(
        new_weekly_raw_data=week_2_raw_data, 
        previous_gdp=official_previous_gdp, 
        current_quarter_pcs=updated_pcs 
    )
    
    if current_prediction is not None:
        print(f">> LIVE NOWCAST (Week 2): {current_prediction:.3f}%")
        print(f"[SYSTEM] Internal Tensor size: {updated_pcs.shape[0]} weeks collected.")
        print("\n=========================================================")
        print("  SYSTEM READY FOR WEEK 3 INGESTION")
        print("=========================================================")