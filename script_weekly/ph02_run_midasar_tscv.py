import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/midas_ready_dataset.csv"
MAX_PCS = 7
INITIAL_TRAIN_QUARTERS = 20

def exponential_almon_weights(theta1, theta2, lags=13):
    w = np.arange(1, lags + 1)
    num = np.exp(np.clip(theta1 * w + theta2 * (w**2), -50, 50))
    return num / np.sum(num)

def midas_ar_loss(params, X_3d, y_lag, y_target, num_pcs):
    """Loss function with the Autoregressive (AR) term added."""
    intercept = params[0]
    ar_coef = params[1]  # The weight given to LAST quarter's GDP
    
    # Start prediction with Intercept + AR term
    preds = np.full(len(y_target), intercept, dtype=float) + (ar_coef * y_lag)
    
    idx = 2
    for p in range(num_pcs):
        gamma = params[idx]
        theta1 = params[idx+1]
        theta2 = params[idx+2]
        idx += 3
        
        weights = exponential_almon_weights(theta1, theta2)
        weighted_x = np.sum(X_3d[:, :, p] * weights, axis=1)
        preds += gamma * weighted_x
        
    return np.sum((y_target - preds)**2)

def evaluate_midasar_dimensions():
    print("--- STEP 1: Restructuring Data for MIDAS-AR ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    X_list, y_list = [], []
    gdp_series = df['GDP_Growth'].dropna()
    
    for target_date, gdp_val in gdp_series.items():
        past_data = df.loc[:target_date]
        if len(past_data) >= 13:
            X_13w = past_data.iloc[-13:][[f'PC{i+1}' for i in range(MAX_PCS)]].values
            X_list.append(X_13w)
            y_list.append(gdp_val)
            
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    
    # Create the AR(1) term: Shift the target variable by 1 quarter
    # We lose the very first quarter because there is no 'previous' GDP for it
    y_target = y_arr[1:]
    y_lag = y_arr[:-1]
    X_arr = X_arr[1:]
    
    print(f"Constructed 3D Tensor: {X_arr.shape[0]} quarters available for AR modeling.")
    
    print("\n--- STEP 2: MIDAS-AR(1) Cross-Validation ---")
    print(f"{'Model Specs':<22} | {'Parameters':<10} | {'Out-of-Sample RMSE'}")
    print("-" * 55)
    
    best_rmse = float('inf')
    optimal_k = 1
    
    for k in range(1, MAX_PCS + 1):
        # Parameters: Intercept (1) + AR coef (1) + (3 per PC)
        num_params = 2 + (k * 3)
        
        if num_params >= INITIAL_TRAIN_QUARTERS:
            print(f"MIDAS-AR with {k} PCs   | {num_params:<10} | OVERFIT (Exceeds DoF)")
            continue
            
        errors = []
        
        for t in range(INITIAL_TRAIN_QUARTERS, len(y_target)):
            X_train, y_lag_train, y_train = X_arr[:t], y_lag[:t], y_target[:t]
            X_test, y_lag_test, y_test = X_arr[t:t+1], y_lag[t:t+1], y_target[t:t+1]
            
            init_params = [np.mean(y_train), 0.5] + [0.0, 0.0, 0.0] * k
            bounds = [(None, None), (-1, 1)] + [(None, None), (-5, 5), (-5, 5)] * k
            
            res = minimize(midas_ar_loss, init_params, args=(X_train, y_lag_train, y_train, k),
                           bounds=bounds, method='L-BFGS-B')
            
            opt_params = res.x
            pred = opt_params[0] + (opt_params[1] * y_lag_test[0])
            
            idx = 2
            for p in range(k):
                gamma, theta1, theta2 = opt_params[idx], opt_params[idx+1], opt_params[idx+2]
                idx += 3
                weights = exponential_almon_weights(theta1, theta2)
                pred += gamma * np.sum(X_test[0, :, p] * weights)
                
            errors.append((y_test[0] - pred)**2)
            
        rmse = np.sqrt(np.mean(errors))
        print(f"MIDAS-AR with {k} PCs   | {num_params:<10} | {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            optimal_k = k

    print("-" * 55)
    print(f"CONCLUSION: The empirically optimal number of components for MIDAS-AR is {optimal_k}.")
    print("Compare this RMSE to the pure MIDAS baseline to prove the AR term's value.")

if __name__ == "__main__":
    evaluate_midasar_dimensions()