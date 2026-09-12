import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

# Suppress optimization warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/midas_ready_dataset.csv"
MAX_PCS = 7
INITIAL_TRAIN_QUARTERS = 20

def exponential_almon_weights(theta1, theta2, lags=13):
    """Generates 13 weekly weights strictly summing to 1."""
    w = np.arange(1, lags + 1)
    # Clip to prevent exponential overflow
    num = np.exp(np.clip(theta1 * w + theta2 * (w**2), -50, 50))
    return num / np.sum(num)

def midas_loss(params, X_3d, y_target, num_pcs):
    """The loss function the optimizer attempts to minimize (Sum of Squared Errors)."""
    intercept = params[0]
    preds = np.full(len(y_target), intercept, dtype=float)
    
    idx = 1
    for p in range(num_pcs):
        gamma = params[idx]
        theta1 = params[idx+1]
        theta2 = params[idx+2]
        idx += 3
        
        weights = exponential_almon_weights(theta1, theta2)
        # Apply the exact weekly weights to the 13 weeks of data
        weighted_x = np.sum(X_3d[:, :, p] * weights, axis=1)
        preds += gamma * weighted_x
        
    return np.sum((y_target - preds)**2)

def evaluate_midas_dimensions():
    print("--- STEP 1: Restructuring Data for MIDAS ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    X_list, y_list = [], []
    gdp_series = df['GDP_Growth'].dropna()
    
    # Pack the data into 13-week blocks corresponding to each GDP release
    for target_date, gdp_val in gdp_series.items():
        past_data = df.loc[:target_date]
        if len(past_data) >= 13:
            X_13w = past_data.iloc[-13:][[f'PC{i+1}' for i in range(MAX_PCS)]].values
            X_list.append(X_13w)
            y_list.append(gdp_val)
            
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    
    print(f"Constructed 3D Tensor: {X_arr.shape[0]} quarters, {X_arr.shape[1]} weeks, {X_arr.shape[2]} PCs.")
    
    print("\n--- STEP 2: Expanding Window Cross-Validation ---")
    print(f"Training starts with {INITIAL_TRAIN_QUARTERS} quarters. Predicting 1-step ahead out-of-sample.")
    print(f"{'Model Specs':<20} | {'Parameters':<10} | {'Out-of-Sample RMSE'}")
    print("-" * 55)
    
    best_rmse = float('inf')
    optimal_k = 1
    
    for k in range(1, MAX_PCS + 1):
        num_params = 1 + (k * 3)
        
        # If parameters exceed available training data, the math breaks
        if num_params >= INITIAL_TRAIN_QUARTERS:
            print(f"MIDAS with {k} PCs    | {num_params:<10} | OVERFIT (Exceeds Degrees of Freedom)")
            continue
            
        errors = []
        
        # The Out-of-Sample Loop
        for t in range(INITIAL_TRAIN_QUARTERS, len(y_arr)):
            X_train, y_train = X_arr[:t], y_arr[:t]
            X_test, y_test = X_arr[t:t+1], y_arr[t:t+1]
            
            # Initial guesses and bounds for optimization stability
            init_params = [np.mean(y_train)] + [0.0, 0.0, 0.0] * k
            bounds = [(None, None)] + [(None, None), (-5, 5), (-5, 5)] * k
            
            res = minimize(midas_loss, init_params, args=(X_train, y_train, k),
                           bounds=bounds, method='L-BFGS-B')
            
            # Predict the unseen quarter
            opt_params = res.x
            pred = opt_params[0]
            idx = 1
            for p in range(k):
                gamma, theta1, theta2 = opt_params[idx], opt_params[idx+1], opt_params[idx+2]
                idx += 3
                weights = exponential_almon_weights(theta1, theta2)
                pred += gamma * np.sum(X_test[0, :, p] * weights)
                
            errors.append((y_test[0] - pred)**2)
            
        rmse = np.sqrt(np.mean(errors))
        print(f"MIDAS with {k} PCs    | {num_params:<10} | {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            optimal_k = k

    print("-" * 55)
    print(f"CONCLUSION: The empirically optimal number of components is {optimal_k}.")
    print(f"This model balances maximum predictive signal with strict degrees of freedom limits.")

if __name__ == "__main__":
    evaluate_midas_dimensions()