import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
MODEL_FILE = "data_weekly/srilanka_gdp_model.pkl"
WEEKLY_DATA_FILE = "data_weekly/master_data_weekly.csv"  # Raw Weekly Keywords
ORIGINAL_DATA_FILE = "data_weekly/final_dataset_ready.csv" # Used to recover PCA map
GDP_FILE = "data_weekly/gdp_sri_lanka.csv" # Actual GDP for plotting

def recover_pca_map():
    """
    Re-learns the PCA definition from the original dataset 
    so 'PC1' means the same thing now as it did during training.
    """
    print("Recovering PCA map...")
    df = pd.read_csv(ORIGINAL_DATA_FILE, index_col=0, parse_dates=True)
    if "GDP_Growth" in df.columns:
        df = df.drop(columns=["GDP_Growth"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    
    return scaler, pca

def run_tracker():
    print("--- STARTING WEEKLY TRACKER ---")
    
    # 1. Load the Saved Model (NO RETRAINING)
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
    
    # 2. Get the PCA Logic
    scaler, pca = recover_pca_map()
    
    # 3. Load & Transform Weekly Data
    print("Processing weekly data...")
    df_weekly = pd.read_csv(WEEKLY_DATA_FILE, index_col=0, parse_dates=True)
    if "GDP_Growth" in df_weekly.columns:
        df_weekly = df_weekly.drop(columns=["GDP_Growth"])
        
    # Transform: Raw -> Scaled -> PCs
    X_weekly_scaled = scaler.transform(df_weekly)
    X_weekly_pcs = pca.transform(X_weekly_scaled)
    
    df_pcs = pd.DataFrame(X_weekly_pcs, columns=['PC1', 'PC2', 'PC3'], index=df_weekly.index)
    
    # 4. Predict
    # The model expects [PC1, PC2, PC3]. We feed it directly.
    # RESULT: This is the "Leading Signal" (GDP forecast for 3 months from now)
    predictions = model.predict(df_pcs)
    
    tracker = pd.DataFrame(index=df_weekly.index)
    tracker['Tracker_Signal'] = predictions
    
    # 5. Visualization Adjustment (The Shift)
    # Since the model was trained with a 1-Quarter Lag (approx 13 weeks),
    # The signal generated TODAY corresponds to GDP 13 weeks in the FUTURE.
    # We shift it forward on the plot so it lines up with the GDP dots.
    tracker['GDP_Forecast_Aligned'] = tracker['Tracker_Signal'].shift(13)
    
    # Smooth (4-week average)
    tracker['GDP_Forecast_Smoothed'] = tracker['GDP_Forecast_Aligned'].rolling(window=4).mean()
    
    # 6. Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot the Tracker
    plt.plot(tracker.index, tracker['GDP_Forecast_Smoothed'], 
             color='#0056b3', linewidth=2, label='Weekly Tracker')
    
    # Plot Actual GDP
    df_gdp = pd.read_csv(GDP_FILE, index_col=0, parse_dates=True)
    # Ensure we use the correct column for GDP
    gdp_col = df_gdp.columns[0] 
    
    plt.plot(df_gdp.index, df_gdp[gdp_col], 
                color='black', linewidth=2, marker='D', zorder=5, label='Actual GDP')
    
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Sri Lanka Weekly GDP Tracker (Using Pre-Fitted Model)")
    plt.ylabel("YoY GDP Growth (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Date Formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig("data_weekly/WEEKLY_TRACKER_RESULT.png")
    tracker.to_csv("data_weekly/final_tracker_values.csv")
    print("Success! Tracker saved to: data_weekly/WEEKLY_TRACKER_RESULT.png")

if __name__ == "__main__":
    run_tracker()