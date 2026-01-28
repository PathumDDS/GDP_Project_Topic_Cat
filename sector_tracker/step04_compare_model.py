import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
INPUT_FILE = "data_weekly/gdp_merged_data.csv"
TARGET_COL = "GDP_Growth"

# The "Refined" Drop List (Noise Filter)
DROP_COLS = [
    "Hobbies___Leisure", "Arts___Entertainment", "Games", 
    "Entertainment_Media", "Entertainment_Industry", 
    "Books___Literature", "Gifts___Special_Event_Items"
]

def run_model_tournament():
    print("--- STEP 1: PREPARING DATA ---")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # 1. Separate Target & Features
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # 2. Filter Noise
    existing_drops = [c for c in DROP_COLS if c in X.columns]
    X_refined = X.drop(columns=existing_drops)
    
    # 3. Lag 0 (Instant Prediction) - Just drop NaNs
    data = pd.concat([X_refined, y], axis=1).dropna()
    X_final = data.iloc[:, :-1]
    y_final = data.iloc[:, -1]
    
    print(f"Data Shape: {X_final.shape} (Quarters, Categories)")

    # --- STEP 2: DEFINING THE CONTENDERS ---
    models = {
        "OLS (Linear Regression)": LinearRegression(),
        "Ridge (L2 Penalty)":      Ridge(alpha=1.0),
        "Lasso (L1 Penalty)":      Lasso(alpha=0.1),
        "ElasticNet (Hybrid)":     ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    # --- STEP 3: THE TOURNAMENT (LOOCV) ---
    print("\n--- STEP 3: RUNNING TOURNAMENT ---")
    results = {}
    
    loo = LeaveOneOut()

    for name, model in models.items():
        actuals = []
        predictions = []
        
        for train_ix, test_ix in loo.split(X_final):
            # A. Split
            X_train, X_test = X_final.iloc[train_ix], X_final.iloc[test_ix]
            y_train, y_test = y_final.iloc[train_ix], y_final.iloc[test_ix]
            
            # B. Preprocessing (Scale + PCA) - INSIDE LOOP to prevent leakage
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_sc)
            X_test_pca = pca.transform(X_test_sc)
            
            # C. Train & Predict
            model.fit(X_train_pca, y_train)
            pred = model.predict(X_test_pca)[0]
            
            actuals.append(y_test.values[0])
            predictions.append(pred)
        
        # Calculate Score
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        results[name] = rmse
        print(f"Finished {name}...")

    # --- STEP 4: DECLARING THE WINNER ---
    print("\n" + "="*30)
    print("FINAL RESULTS (RMSE - Lower is Better)")
    print("="*30)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}: {score:.4f}")
        
    winner = sorted_results[0][0]
    print("="*30)
    print(f">> RECOMMENDATION: Use {winner}")

    # Optional: Visualization of the comparison
    names = [x[0].split()[0] for x in sorted_results] # Short names
    scores = [x[1] for x in sorted_results]
    
    plt.figure(figsize=(8, 4))
    plt.bar(names, scores, color=['green', 'blue', 'blue', 'blue'])
    plt.ylabel('RMSE (Lower is better)')
    plt.title('Model Comparison Tournament')
    plt.ylim(0, max(scores) * 1.1)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    plt.savefig("data_weekly/model_comparison.png")
    print("Comparison chart saved to data_weekly/model_comparison.png")

if __name__ == "__main__":
    run_model_tournament()