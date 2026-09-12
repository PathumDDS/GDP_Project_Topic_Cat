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

def midas_ar_dummy_loss(params, X_3d, y_lag, d_crisis, y_target, num_pcs):
    """Loss function with AR term AND a Crisis Dummy Variable."""
    intercept = params[0]
    ar_coef = params[1]
    dummy_coef = params[2]  # The weight applied ONLY during the 2022/2023 crisis
    
    # Base prediction: Intercept + AR momentum + Crisis Adjustment
    preds = np.full(len(y_target), intercept, dtype=float) + (ar_coef * y_lag) + (dummy_coef * d_crisis)
    
    idx = 3
    for p in range(num_pcs):
        gamma = params[idx]
        theta1 = params[idx+1]
        theta2 = params[idx+2]
        idx += 3
        
        weights = exponential_almon_weights(theta1, theta2)
        weighted_x = np.sum(X_3d[:, :, p] * weights, axis=1)
        preds += gamma * weighted_x
        
    return np.sum((y_target - preds)**2)

def evaluate_midasar_dummy_dimensions():
    print("--- STEP 1: Restructuring Data for MIDAS-AR with Structural Break ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    X_list, y_list, d_list = [], [], []
    gdp_series = df['GDP_Growth'].dropna()
    
    for target_date, gdp_val in gdp_series.items():
        past_data = df.loc[:target_date]
        if len(past_data) >= 13:
            X_13w = past_data.iloc[-13:][[f'PC{i+1}' for i in range(MAX_PCS)]].values
            X_list.append(X_13w)
            y_list.append(gdp_val)
            
            # The Structural Break: 1.0 during the crisis years, 0.0 otherwise
            if target_date.year in [2022, 2023]:
                d_list.append(1.0)
            else:
                d_list.append(0.0)
            
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    d_arr = np.array(d_list)
    
    # Shift AR terms
    y_target = y_arr[1:]
    y_lag = y_arr[:-1]
    X_arr = X_arr[1:]
    d_target = d_arr[1:]  # Dummy corresponds to the target quarter
    
    print(f"Constructed 3D Tensor: {X_arr.shape[0]} quarters available.")
    
    print("\n--- STEP 2: MIDAS-AR(1) + Crisis Dummy Cross-Validation ---")
    print(f"{'Model Specs':<26} | {'Parameters':<10} | {'Out-of-Sample RMSE'}")
    print("-" * 60)
    
    best_rmse = float('inf')
    optimal_k = 1
    
    for k in range(1, MAX_PCS + 1):
        # Parameters: Intercept (1) + AR coef (1) + Dummy coef (1) + (3 per PC)
        num_params = 3 + (k * 3)
        
        if num_params >= INITIAL_TRAIN_QUARTERS:
            print(f"MIDAS-AR+Dummy with {k} PCs  | {num_params:<10} | OVERFIT (Exceeds DoF)")
            continue
            
        errors = []
        
        for t in range(INITIAL_TRAIN_QUARTERS, len(y_target)):
            X_train, y_lag_train = X_arr[:t], y_lag[:t]
            d_train, y_train = d_target[:t], y_target[:t]
            
            X_test, y_lag_test = X_arr[t:t+1], y_lag[t:t+1]
            d_test, y_test = d_target[t:t+1], y_target[t:t+1]
            
            # Initial guesses: [Intercept, AR, Dummy, ... PCs]
            init_params = [np.mean(y_train), 0.5, 0.0] + [0.0, 0.0, 0.0] * k
            bounds = [(None, None), (-1, 1), (None, None)] + [(None, None), (-5, 5), (-5, 5)] * k
            
            res = minimize(midas_ar_dummy_loss, init_params, args=(X_train, y_lag_train, d_train, y_train, k),
                           bounds=bounds, method='L-BFGS-B')
            
            opt_params = res.x
            pred = opt_params[0] + (opt_params[1] * y_lag_test[0]) + (opt_params[2] * d_test[0])
            
            idx = 3
            for p in range(k):
                gamma, theta1, theta2 = opt_params[idx], opt_params[idx+1], opt_params[idx+2]
                idx += 3
                weights = exponential_almon_weights(theta1, theta2)
                pred += gamma * np.sum(X_test[0, :, p] * weights)
                
            errors.append((y_test[0] - pred)**2)
            
        rmse = np.sqrt(np.mean(errors))
        print(f"MIDAS-AR+Dummy with {k} PCs  | {num_params:<10} | {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            optimal_k = k

    print("-" * 60)
    print(f"CONCLUSION: The empirically optimal number of components is {optimal_k}.")

if __name__ == "__main__":
    evaluate_midasar_dummy_dimensions()