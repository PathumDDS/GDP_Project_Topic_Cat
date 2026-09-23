import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/midas_ready_dataset.csv"
PLOT_PREDICTIONS = "data_weekly/nowcast_predictions_by_quarter.png"
PLOT_ERRORS = "data_weekly/nowcast_errors_by_quarter.png"
WEIGHTS_OUTPUT_FILE = "models/midas_optimized_weights.npy"
OPTIMAL_PCS = 3  
HOLDOUT_YEAR = 2025  

def exponential_almon_weights(theta1, theta2, lags):
    w = np.arange(1, lags + 1)
    num = np.exp(np.clip(theta1 * w + theta2 * (w**2), -50, 50))
    return num / np.sum(num)

def midas_ar_dummy_loss_weekly(params, X_3d, y_lag, d_crisis, y_target, num_pcs, num_weeks):
    intercept, ar_coef, dummy_coef = params[0], params[1], params[2]
    preds = np.full(len(y_target), intercept, dtype=float) + (ar_coef * y_lag) + (dummy_coef * d_crisis)
    
    idx = 3
    for p in range(num_pcs):
        gamma, theta1, theta2 = params[idx], params[idx+1], params[idx+2]
        idx += 3
        weights = exponential_almon_weights(theta1, theta2, num_weeks)
        weighted_x = np.sum(X_3d[:, :num_weeks, p] * weights, axis=1)
        preds += gamma * weighted_x
        
    return np.sum((y_target - preds)**2)

def run_comprehensive_evaluation():
    print("--- STEP 1: Restructuring Data ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    X_list, y_list, d_list, date_list = [], [], [], []
    gdp_series = df['GDP_Growth'].dropna()
    
    for target_date, gdp_val in gdp_series.items():
        past_data = df.loc[:target_date]
        if len(past_data) >= 13:
            X_13w = past_data.iloc[-13:][[f'PC{i+1}' for i in range(OPTIMAL_PCS)]].values
            X_list.append(X_13w)
            y_list.append(gdp_val)
            date_list.append(target_date)
            d_list.append(1.0 if target_date.year in [2022, 2023] else 0.0)
            
    X_arr, y_arr, d_arr, dates_arr = np.array(X_list), np.array(y_list), np.array(d_list), np.array(date_list)[1:]
    y_target, y_lag, X_arr, d_target = y_arr[1:], y_arr[:-1], X_arr[1:], d_arr[1:]
    
    train_mask = np.array([d.year < HOLDOUT_YEAR for d in dates_arr])
    test_mask = np.array([d.year == HOLDOUT_YEAR for d in dates_arr])
    
    X_train, y_lag_train, d_train, y_train = X_arr[train_mask], y_lag[train_mask], d_target[train_mask], y_target[train_mask]
    X_test, y_lag_test, d_test, y_test = X_arr[test_mask], y_lag[test_mask], d_target[test_mask], y_target[test_mask]
    
    num_test_quarters = min(3, len(y_test))
    
    print(f"\n--- STEP 2: Tracking Predictions & Errors for {HOLDOUT_YEAR} ---")
    target_str = " | ".join([f"Q{i+1}: {y_test[i]:.2f}%" for i in range(num_test_quarters)])
    print(f"Target GDPs -> {target_str}\n")
    
    preds_dict = {i: [] for i in range(num_test_quarters)}
    errors_dict = {i: [] for i in range(num_test_quarters)}
    
    for num_weeks in range(1, 14):
        init_params = [np.mean(y_train), 0.5, 0.0] + [0.0, 0.0, 0.0] * OPTIMAL_PCS
        bounds = [(None, None), (-1, 1), (None, None)] + [(None, None), (-5, 5), (-5, 5)] * OPTIMAL_PCS
        
        res = minimize(midas_ar_dummy_loss_weekly, init_params, 
                       args=(X_train, y_lag_train, d_train, y_train, OPTIMAL_PCS, num_weeks),
                       bounds=bounds, method='L-BFGS-B')
        
        opt_params = res.x
        preds = np.full(len(y_test), opt_params[0]) + (opt_params[1] * y_lag_test) + (opt_params[2] * d_test)
        
        idx = 3
        for p in range(OPTIMAL_PCS):
            gamma, theta1, theta2 = opt_params[idx], opt_params[idx+1], opt_params[idx+2]
            idx += 3
            weights = exponential_almon_weights(theta1, theta2, num_weeks)
            preds += gamma * np.sum(X_test[:, :num_weeks, p] * weights, axis=1)
            
        print_str = f"Week {num_weeks:2d} |"
        for i in range(num_test_quarters):
            preds_dict[i].append(preds[i])
            errors_dict[i].append(abs(y_test[i] - preds[i]))
            print_str += f" Q{i+1} Pred: {preds[i]:>5.2f}% (Err: {errors_dict[i][-1]:.2f}) |"
            
        print(print_str)

        # --- EXPORT WEIGHTS AT FULL HORIZON (WEEK 13) ---
        if num_weeks == 13:
            os.makedirs("models", exist_ok=True)
            np.save(WEIGHTS_OUTPUT_FILE, opt_params)
            print(f"\n[SUCCESS] Exported final Week 13 MIDAS weights to '{WEIGHTS_OUTPUT_FILE}'")

    # --- STEP 3: PREDICTION PLOT ---
    fig1, axes1 = plt.subplots(1, num_test_quarters, figsize=(6 * num_test_quarters, 5))
    if num_test_quarters == 1: axes1 = [axes1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(num_test_quarters):
        axes1[i].plot(range(1, 14), preds_dict[i], marker='o', linestyle='-', color=colors[i], linewidth=2, label='Weekly Nowcast')
        axes1[i].axhline(y_test[i], color='red', linestyle='--', linewidth=2, label='Actual GDP')
        axes1[i].set_title(f"{HOLDOUT_YEAR} Q{i+1} GDP Nowcast", fontsize=13, fontweight='bold')
        axes1[i].set_xlabel("Weeks of Google Trends Data", fontsize=11)
        axes1[i].set_xticks(range(1, 14))
        axes1[i].grid(True, linestyle='--', alpha=0.6)
        axes1[i].legend(loc='best')
        
    axes1[0].set_ylabel("GDP Growth Rate (%)", fontsize=12)
    fig1.suptitle("Intra-Quarter Nowcasting: Model Convergence on Actual GDP", fontsize=16, fontweight='bold', y=1.05)
    
    fig1.tight_layout()
    fig1.savefig(PLOT_PREDICTIONS, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- STEP 4: ERROR PLOT ---
    fig2, axes2 = plt.subplots(1, num_test_quarters, figsize=(6 * num_test_quarters, 5), sharey=True)
    if num_test_quarters == 1: axes2 = [axes2]
    
    for i in range(num_test_quarters):
        axes2[i].plot(range(1, 14), errors_dict[i], marker='s', linestyle='-', color=colors[i], linewidth=2)
        axes2[i].set_title(f"{HOLDOUT_YEAR} Q{i+1}: Absolute Error", fontsize=13, fontweight='bold')
        axes2[i].set_xlabel("Weeks of Google Trends Data", fontsize=11)
        axes2[i].set_xticks(range(1, 14))
        axes2[i].grid(True, linestyle='--', alpha=0.6)
        
    axes2[0].set_ylabel("Absolute Error (Percentage Points)", fontsize=12)
    fig2.suptitle("Intra-Quarter Nowcasting: Weekly Error Trajectory", fontsize=16, fontweight='bold', y=1.05)
    
    fig2.tight_layout()
    fig2.savefig(PLOT_ERRORS, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"\nSUCCESS: Plots saved to:")
    print(f"1. {PLOT_PREDICTIONS}")
    print(f"2. {PLOT_ERRORS}")

if __name__ == "__main__":
    run_comprehensive_evaluation()