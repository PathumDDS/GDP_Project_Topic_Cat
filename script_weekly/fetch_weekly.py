# script_weekly/fetch_weekly.py
# FIXED VERSION
# - Added random sleep between successful requests
# - Added handling for "<1" string values from Google
# - Switched to Mean-based scaling (safer for sparse data)

import os, time, random, traceback
import pandas as pd
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq
# Add 'timezone' to the imports
from datetime import datetime, timedelta, timezone


# ----------------- Paths -----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KW_DIR = os.path.join(ROOT, "keywords_weekly")
RAW_WINDOWS = os.path.join(ROOT, "data_weekly", "raw_windows")
RAW_WEEKLY = os.path.join(ROOT, "data_weekly", "raw_weekly")
LOGS = os.path.join(ROOT, "logs_weekly")

os.makedirs(LOGS, exist_ok=True)
os.makedirs(RAW_WINDOWS, exist_ok=True)
os.makedirs(RAW_WEEKLY, exist_ok=True)

UNPRO = os.path.join(KW_DIR, "unprocessed.txt")
PROCING = os.path.join(KW_DIR, "processing.txt")
PROCED = os.path.join(KW_DIR, "processed.txt")
FAILED = os.path.join(KW_DIR, "failed.txt")
RUN_LOG = os.path.join(LOGS, "runs.log")

GEO = "LK"
TZ = 330

# ----------------- Configuration -----------------
WINDOW_YEARS = 5
STEP_YEARS = 4
START_DATE = datetime(2015, 1, 1)

# RETRY SETTINGS
MAX_RETRIES = 3 
# Random sleep range (seconds) to mimic human behavior
MIN_SLEEP = 30
MAX_SLEEP = 60

# ----------------- Helpers -----------------
def log(msg):
    # OLD: ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    # NEW:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} - {msg}\n")
    print(msg)

def append_line(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_lines(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]

def pop_keyword():
    lines = read_lines(UNPRO)
    if not lines: return None
    first = lines[0]
    rest = lines[1:]
    with open(UNPRO, "w", encoding="utf-8") as f:
        for r in rest: f.write(r + "\n")
    return first

def save_status_move(keyword, target):
    if os.path.exists(PROCING):
        lines = read_lines(PROCING)
        lines = [l for l in lines if l != keyword]
        with open(PROCING, "w", encoding="utf-8") as f:
            for l in lines: f.write(l + "\n")
    append_line(target, keyword)

def sanitize_for_filename(name):
    s = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)
    s = s.strip().replace(" ", "_")
    return s if s else "keyword"

# ----------------- Data Cleaning Helper -----------------
def clean_google_data(df):
    """Converts '<1' to 0 and ensures numeric types"""
    for col in df.columns:
        # If column is object type, it might contain '<1'
        if df[col].dtype == object:
            df[col] = df[col].replace({'<1': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# ----------------- Fetch Logic -----------------
def fetch_window(pytrends, kw_search, start, end, safe_kw):
    start_adj = start - timedelta(days=(start.weekday() + 1) % 7) 
    end_adj = end + timedelta(days=(6 - end.weekday()) % 7)
    timeframe = f"{start_adj:%Y-%m-%d} {end_adj:%Y-%m-%d}"

    full_idx = pd.date_range(start_adj, end_adj, freq="W-SUN")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Random jitter before request (except first attempt)
            if attempt > 1:
                sleep_time = random.randint(MIN_SLEEP, MAX_SLEEP) + (attempt * 5)
                log(f"Retry {attempt}/{MAX_RETRIES} for {start.date()}... sleeping {sleep_time}s")
                time.sleep(sleep_time)

            pytrends.build_payload([kw_search], timeframe=timeframe, geo=GEO)
            df = pytrends.interest_over_time()

            # --- NEW: If Google returned valid but empty/zero data, DO NOT RETRY ---
            if df is not None and not df.empty:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

                df = clean_google_data(df)

                real_col = df.columns[0]
                if df[real_col].sum() == 0:
                    # Structural zero → return NA window, no retry
                    return pd.DataFrame(index=full_idx, columns=[safe_kw])


            if df is None or df.empty:
                return pd.DataFrame(index=full_idx, columns=[safe_kw])

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            # Clean '<1' strings
            df = clean_google_data(df)

            real_cols = [c for c in df.columns]
            if not real_cols:
                return pd.DataFrame(index=full_idx, columns=[safe_kw])

            real_col = real_cols[0]
            df = df.rename(columns={real_col: safe_kw})
            df = df.reindex(full_idx)
            
            return df

        except Exception as ex:
            log(f"Error fetching {start.date()}: {ex}")
            
            # --- CHANGE C: The Penalty Box ---
            if "429" in str(ex) or "Too Many Requests" in str(ex):
                log("⚠️ HIT RATE LIMIT (429). Sleeping 10 minutes to cool down...")
                time.sleep(600)  # Sleep 10 minutes (600 seconds)
            else:
                time.sleep(60)   # Normal short sleep

    return pd.DataFrame(index=full_idx, columns=[safe_kw])

# ----------------- Window Computation -----------------
def compute_windows():
    windows = []
    cur = START_DATE
    today = datetime.utcnow()

    # First day of current month
    first_day_this_month = today.replace(day=1)

    # Last day of previous month
    last_day_prev_month = first_day_this_month - timedelta(days=1)

    # Align to last completed Sunday
    now = last_day_prev_month - timedelta(
        days=(last_day_prev_month.weekday() - 6) % 7
    )

    
    while cur + relativedelta(years=WINDOW_YEARS) <= now:
        windows.append((cur, cur + relativedelta(years=WINDOW_YEARS)))
        cur = cur + relativedelta(years=STEP_YEARS) # 1 year overlap

    if (now - cur).days >= 7:
        windows.append((cur, now))
    return windows

# ----------------- Robust Stitching -----------------
def stitch_windows(windows_list):
    if not windows_list: return None

    # Start with the first window
    stitched = windows_list[0][0].copy().sort_index()

    for i in range(1, len(windows_list)):
        prev_df, prev_s, prev_e = windows_list[i - 1]
        df, s, e = windows_list[i]

        # Define Overlap Range
        overlap_start = max(prev_s, s)
        # Ensure we don't go past the end of the previous window in the overlap calculation
        overlap_end = min(prev_e, e)

        # Get the overlapping slices
        overlap_old = stitched.loc[overlap_start:overlap_end]
        overlap_new = df.loc[overlap_start:overlap_end]

        scale = 1.0
        
        # MEAN based scaling (More robust than median for sparse data)
        # We use a small epsilon to avoid division by zero
        mean_old = overlap_old.mean().iloc[0]
        mean_new = overlap_new.mean().iloc[0]

        if mean_new > 0 and mean_old > 0:
            scale = mean_old / mean_new
        elif mean_new == 0 and mean_old > 0:
            # If new data is all zeros but old data wasn't, we can't scale nicely.
            # Fallback: keep scale 1.0 or treat as effectively zero.
            scale = 1.0 
        
        df_scaled = df * scale
        
        # Append only the new data (post-overlap)
        # We cut off the new dataframe to start AFTER the previous window ended
        tail_start = prev_e + timedelta(days=1)
        tail = df_scaled.loc[tail_start:]

        stitched = pd.concat([stitched, tail])

    # Final cleanup
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    return stitched.sort_index()

# ----------------- Main (Fixed: Headers + Win_list) -----------------
def main():
    while True:
        keyword = pop_keyword()
        
        if not keyword:
            log("All keywords processed. Exiting.")
            break

        save_status_move(keyword, PROCING)
        safe_kw = sanitize_for_filename(keyword)
        log(f"--- START: {keyword} ---")

        # 1. Define Fake Browser Headers (Fixes 429 Errors)
        fake_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        }

        # 2. Initialize PyTrends with headers and NO SSL verify
        pytrends = TrendReq(
            hl="en-US", 
            tz=TZ, 
            retries=2, 
            backoff_factor=0.1, 
            requests_args={'verify': False, 'headers': fake_headers}
        )

        # 3. COMPUTE WINDOWS (This was the missing line!)
        win_list = compute_windows()
        
        collected = []
        non_empty_count = 0
        failed_keyword = False

        # 4. Loop through windows
        for idx, (s, e) in enumerate(win_list):
            if idx > 0:
                wait = random.randint(MIN_SLEEP, MAX_SLEEP)
                log(f"Window Sleep: {wait}s...")
                time.sleep(wait)

            df = fetch_window(pytrends, keyword.strip(), s, e, safe_kw)
            
            if safe_kw in df.columns and df[safe_kw].sum() > 0:
                non_empty_count += 1
            
            collected.append((df, s, e))

        # 5. Handle Results
        if non_empty_count == 0:
            log(f"FAIL: No data found for {keyword}")
            save_status_move(keyword, FAILED)
            failed_keyword = True
        
        if not failed_keyword:
            stitched = stitch_windows(collected)
            
            # Save Raw Windows
            win_dir = os.path.join(RAW_WINDOWS, safe_kw)
            os.makedirs(win_dir, exist_ok=True)
            for (df, s, e) in collected:
                fname = f"{safe_kw}_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.csv"
                df.to_csv(os.path.join(win_dir, fname))

            # Save Final
            outpath = os.path.join(RAW_WEEKLY, f"{safe_kw}_weekly.csv")
            stitched.to_csv(outpath)
            log(f"SUCCESS: Saved {outpath}")
            save_status_move(keyword, PROCED)

        # 6. Batch Sleep (Prevent Ban)
        long_sleep = random.randint(60, 120)
        log(f"=== Keyword Finished. Sleeping {long_sleep}s before next keyword... ===")
        time.sleep(long_sleep)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CRITICAL FAILURE: {e}")
        traceback.print_exc()