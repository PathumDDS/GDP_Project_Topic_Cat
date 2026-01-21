# Weekly GDP Tracker: Sri Lanka & Regional Peers

This repository contains the Python implementation of a **Weekly Economic Tracker** for Sri Lanka (and extending to India), based on the OECD "Weekly Tracker" methodology developed by Nicolas Woloszko.

The system uses Google Trends search data to nowcast quarterly GDP growth on a weekly frequency, bridging the gap between official data releases.

## 📊 Methodology

1.  **Data Extraction**: Weekly search volume data is fetched from Google Trends using a "sliding window" approach to ensure high-frequency resolution over long periods (2015–Present).
2.  **Dimensionality Reduction**: Principal Component Analysis (PCA) is applied to ~200 economic keywords (e.g., "unemployment," "tourism," "loans") to extract latent economic signals.
3.  **Modeling**: A Linear Regression model (trained on quarterly aggregated signals) relates search behavior at time $T-1$ to GDP growth at time $T$.
4.  **Forecasting**: The fitted model predicts weekly economic activity, providing a leading indicator for GDP growth.

## 📂 Project Structure

```text
├── data_weekly/           # (Ignored by Git) Stores raw CSVs and processed datasets
├── keywords_weekly/       # Contains lists of keywords/categories to search
├── logs_weekly/           # Execution logs for the scraper
├── script_weekly/         # Main scripts
│   ├── fetch_categories.py    # Downloads Google Trends data for Sri Lanka/India
│   └── ...
├── run_weekly_tracker.py  # Main execution script: Loads model & generates plots
├── .gitignore             # Ensures large data files are not uploaded
└── README.md              # Project documentation