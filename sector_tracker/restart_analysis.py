import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/gdp_merged_data.csv"
TARGET_COL = "GDP_Growth"
OUTLIER_THRESHOLD = 10.0
N_COMPONENTS = 2

def run_final_analysis():
    print("--- STEP 1: DATA LOADING & OUTLIER REMOVAL ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    data = pd.concat([X, y], axis=1).dropna()
    
    # Filter Outlier
    mask = np.abs(data[y.name]) <= OUTLIER_THRESHOLD
    data_clean = data[mask]
    n_dropped = len(data) - len(data_clean)
    
    print(f"Categories: {X.shape[1]} (All 16 retained)")
    print(f"Quarters:   {len(data_clean)} (Dropped {n_dropped} outlier)")
    
    X_clean = data_clean.iloc[:, :-1]
    y_clean = data_clean.iloc[:, -1]

    # --- STEP 2: GENERATING SCREE PLOT ---
    print("\n--- STEP 2: GENERATING SCREE PLOT ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    pca = PCA()
    pca.fit(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), cum_var[:10], 'bo-', linewidth=2, markersize=8)
    plt.axhline(0.80, color='r', linestyle='--', label='80% Explained Variance')
    plt.bar(range(1, 11), pca.explained_variance_ratio_[:10], alpha=0.3, label='Individual Variance')
    plt.title(f"Scree Plot: Variance of 16 Categories (Cleaned Data)")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("data_weekly/final_scree_plot.png")
    print(">> Saved: data_weekly/final_scree_plot.png")

    # --- STEP 3: FIND BEST MODEL (Tournament) ---
    print("\n--- STEP 3: RUNNING MODEL TOURNAMENT ---")
    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    best_name = None
    best_rmse = float('inf')
    best_preds = []
    
    loo = LeaveOneOut()
    
    # Store predictions for the final plot
    actuals_for_plot = [] 
    dates_for_plot = []

    for name, model in models.items():
        current_preds = []
        current_actuals = []
        
        # Cross-Validation Loop
        for tr, te in loo.split(X_clean):
            X_tr, X_te = X_clean.iloc[tr], X_clean.iloc[te]
            y_tr, y_te = y_clean.iloc[tr], y_clean.iloc[te]
            
            # 1. Scale
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            
            # 2. PCA
            p = PCA(n_components=N_COMPONENTS)
            X_tr_p = p.fit_transform(X_tr_s)
            X_te_p = p.transform(X_te_s)
            
            # 3. Fit/Predict
            model.fit(X_tr_p, y_tr)
            pred = model.predict(X_te_p)[0]
            
            current_preds.append(pred)
            current_actuals.append(y_te.values[0]) # FIXED: Always append actuals
            
            # Save dates/actuals ONCE for plotting later
            if name == "OLS": 
                dates_for_plot.append(X_clean.index[te[0]])
                actuals_for_plot.append(y_te.values[0])

        # Score this model
        rmse = np.sqrt(mean_squared_error(current_actuals, current_preds))
        print(f"  {name} RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_preds = current_preds

    print(f"\n>> WINNER: {best_name} (RMSE: {best_rmse:.4f})")

    # --- STEP 4: ACTUAL VS PREDICTED PLOT ---
    print("\n--- STEP 4: GENERATING FINAL PLOT ---")
    
    res_df = pd.DataFrame({
        'Actual': actuals_for_plot,
        'Predicted': best_preds
    }, index=dates_for_plot).sort_index()
    
    r2 = r2_score(res_df['Actual'], res_df['Predicted'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(res_df.index, res_df['Actual'], 'k-', marker='o', label='Actual GDP Growth', linewidth=1.5)
    plt.plot(res_df.index, res_df['Predicted'], 'b--', marker='x', label=f'Tracker ({best_name})', linewidth=1.5)
    
    plt.axhline(0, color='r', linestyle='-', linewidth=0.5)
    plt.title(f"Other Personal Services Tracker\nModel: {best_name} | RMSE: {best_rmse:.2f} | R²: {r2:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    plt.savefig("data_weekly/final_actual_vs_predicted.png")
    print(">> Saved: data_weekly/final_actual_vs_predicted.png")
    
    res_df.to_csv("data_weekly/final_tracker_results.csv")

if __name__ == "__main__":
    run_final_analysis()