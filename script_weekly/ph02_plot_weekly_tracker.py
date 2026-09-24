import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

def exponential_almon_weights(theta1, theta2, lags=13):
    w = np.arange(1, lags + 1)
    num = np.exp(np.clip(theta1 * w + theta2 * (w**2), -50, 50))
    return num / np.sum(num)

def generate_weekly_tracker():
    print("--- Generating Continuous Weekly GDP Tracker ---")
    
    # 1. Load the data and the optimized frozen weights
    df = pd.read_csv("data_weekly/midas_ready_dataset.csv", index_col=0, parse_dates=True)
    opt_params = np.load("models/midas_optimized_weights.npy")
    
    intercept, ar_coef, dummy_coef = opt_params[0], opt_params[1], opt_params[2]
    
    # 2. Extract official GDP and create the AR(1) lagged series
    gdp_data = df['GDP_Growth'].dropna()
    actual_dates = gdp_data.index
    actual_vals = gdp_data.values
    lagged_gdp = gdp_data.shift(1) # Shift to get the previous quarter's GDP
    
    tracker_dates = []
    tracker_preds = []
    
    # 3. Slide a 13-week window across the entire timeline (week by week)
    for i in range(13, len(df)):
        current_date = df.index[i]
        
        # Extract the last 13 weeks of PCA scores ending at this current week
        X_window = df.iloc[i-12:i+1][['PC1', 'PC2', 'PC3']].values
        
        # Find the most recently available Lagged GDP for the AR(1) term
        past_gdp = lagged_gdp[lagged_gdp.index <= current_date]
        if len(past_gdp) == 0 or pd.isna(past_gdp.iloc[-1]):
            continue
        current_y_lag = past_gdp.iloc[-1]
        
        # Apply the Structural Crisis Dummy
        d_crisis = 1.0 if current_date.year in [2022, 2023] else 0.0
        
        # Calculate Base Prediction (Intercept + AR(1) + Dummy)
        pred = intercept + (ar_coef * current_y_lag) + (dummy_coef * d_crisis)
        
        # Apply the frozen Almon lag weights to the high-frequency PC window
        idx = 3
        for p in range(3):
            gamma, theta1, theta2 = opt_params[idx], opt_params[idx+1], opt_params[idx+2]
            idx += 3
            weights = exponential_almon_weights(theta1, theta2, lags=13)
            pred += gamma * np.sum(X_window[:, p] * weights)
            
        tracker_dates.append(current_date)
        tracker_preds.append(pred)

    # 4. Plotting (Stylized exactly like the supervisor's request)
    plt.figure(figsize=(14, 6))
    
    # Plot the continuous weekly tracker
    plt.plot(tracker_dates, tracker_preds, color='#1f77b4', linewidth=2, label='Weekly Tracker (MIDAS-AR)')
    
    # Plot the actual official GDP points
    plt.plot(actual_dates, actual_vals, color='black', marker='D', markersize=6, linestyle='-', linewidth=1.5, label='Actual GDP (Quarterly)')
    
    # Formatting
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.6) # Zero-growth line
    plt.title('Sri Lanka High-Frequency Weekly GDP Tracker (2016-2025)', fontsize=14, fontweight='bold')
    plt.ylabel('YoY GDP Growth (%)', fontsize=12)
    
    # X-axis date formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(loc='upper right', framealpha=1.0)
    plt.tight_layout()
    
    # Save the figure
    output_file = "data_weekly/sri_lanka_weekly_tracker.png"
    plt.savefig(output_file, dpi=300)
    print(f"-> SUCCESS: Weekly tracker plot saved to '{output_file}'")
    plt.show()

if __name__ == "__main__":
    generate_weekly_tracker()