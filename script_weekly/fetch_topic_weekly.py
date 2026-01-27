# script_weekly/fetch_weekly.py
# TOPIC SUGGESTION VERSION
# - Automatically resolves Topic IDs from strings
# - Maintains original window-stitching and file structure

import os, time, random, traceback
import pandas as pd
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq

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
START_DATE = datetime(2015, 1, 1)
WINDOW_YEARS = 5
STEP_YEARS = 4
MAX_RETRIES = 3 
MIN_SLEEP = 30
MAX_SLEEP = 60

# ----------------- Helpers -----------------
def log(msg):
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

def clean_google_data(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({'<1': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# ----------------- Fetch Logic -----------------
def fetch_window(pytrends, topic_id, start, end, original_kw):
    start_adj = start - timedelta(days=(start.weekday() + 1) % 7) 
    end_adj = end + timedelta(days=(6 - end.weekday()) % 7)
    timeframe = f"{start_adj:%Y-%m-%d} {end_adj:%Y-%m-%d}"
    full_idx = pd.date_range(start_adj, end_adj, freq="W-SUN")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                sleep_time = random.randint(MIN_SLEEP, MAX_SLEEP) + (attempt * 5)
                log(f"Retry {attempt}/{MAX_RETRIES}... sleeping {sleep_time}s")
                time.sleep(sleep_time)

            # Build payload using the resolved Topic ID (MID)
            pytrends.build_payload([topic_id], timeframe=timeframe, geo=GEO)
            df = pytrends.interest_over_time()

            if df is not None and not df.empty:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                df = clean_google_data(df)

                # Rename the column from the MID back to your original keyword
                if topic_id in df.columns:
                    df = df.rename(columns={topic_id: original_kw})
                
                if df[original_kw].sum() == 0:
                    return pd.DataFrame(index=full_idx, columns=[original_kw])

                return df.reindex(full_idx)
            return pd.DataFrame(index=full_idx, columns=[original_kw])

        except Exception as ex:
            log(f"Error fetching: {ex}")
            if "429" in str(ex):
                log("Rate Limit Hit. Cooling down 10 mins...")
                time.sleep(600)
            else:
                time.sleep(60)
    return pd.DataFrame(index=full_idx, columns=[original_kw])

def compute_windows():
    windows = []
    cur = START_DATE
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    last_day_prev_month = today.replace(day=1) - timedelta(days=1)
    now = last_day_prev_month - timedelta(days=(last_day_prev_month.weekday() - 6) % 7)
    while cur + relativedelta(years=WINDOW_YEARS) <= now:
        windows.append((cur, cur + relativedelta(years=WINDOW_YEARS)))
        cur = cur + relativedelta(years=STEP_YEARS)
    if (now - cur).days >= 7:
        windows.append((cur, now))
    return windows

def stitch_windows(windows_list, kw_name):
    if not windows_list: return None
    stitched = windows_list[0][0].copy().sort_index()
    for i in range(1, len(windows_list)):
        prev_df, prev_s, prev_e = windows_list[i - 1]
        df, s, e = windows_list[i]
        overlap_start, overlap_end = max(prev_s, s), min(prev_e, e)
        scale = 1.0
        m_old = stitched.loc[overlap_start:overlap_end].mean().iloc[0]
        m_new = df.loc[overlap_start:overlap_end].mean().iloc[0]
        if m_new > 0 and m_old > 0: scale = m_old / m_new
        df_scaled = df * scale
        stitched = pd.concat([stitched, df_scaled.loc[prev_e + timedelta(days=1):]])
    return stitched[~stitched.index.duplicated(keep="last")].sort_index()

# ----------------- Main -----------------
def main():
    while True:
        keyword = pop_keyword()
        if not keyword: break
        save_status_move(keyword, PROCING)
        safe_kw = sanitize_for_filename(keyword)
        log(f"--- RESOLVING TOPIC: {keyword} ---")

        pytrends = TrendReq(hl="en-US", tz=TZ, requests_args={'verify': False})

        # --- STEP 1: RESOLVE TOPIC ID ---
        try:
            suggestions = pytrends.suggestions(keyword)
            if not suggestions:
                log(f"FAIL: No suggestions found for {keyword}")
                save_status_move(keyword, FAILED)
                continue
            
            topic_id = suggestions[0]['mid']
            log(f"Found MID: {topic_id} for {keyword}")
        except Exception as e:
            log(f"Error resolving suggestion for {keyword}: {e}")
            save_status_move(keyword, FAILED)
            continue

        # --- STEP 2: FETCH USING TOPIC ID ---
        win_list = compute_windows()
        collected = []
        non_empty = 0

        for idx, (s, e) in enumerate(win_list):
            if idx > 0: time.sleep(random.randint(MIN_SLEEP, MAX_SLEEP))
            df = fetch_window(pytrends, topic_id, s, e, keyword)
            if keyword in df.columns and df[keyword].sum() > 0:
                non_empty += 1
            collected.append((df, s, e))

        if non_empty == 0:
            save_status_move(keyword, FAILED)
        else:
            stitched = stitch_windows(collected, keyword)
            win_dir = os.path.join(RAW_WINDOWS, safe_kw)
            os.makedirs(win_dir, exist_ok=True)
            for (df, s, e) in collected:
                fname = f"{safe_kw}_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.csv"
                df.to_csv(os.path.join(win_dir, fname))
            
            outpath = os.path.join(RAW_WEEKLY, f"{safe_kw}_weekly.csv")
            stitched.to_csv(outpath)
            save_status_move(keyword, PROCED)
            log(f"SUCCESS: {keyword}")

        # Check if more keywords exist before sleeping ---
        remaining = read_lines(UNPRO)
        if remaining:
            # If there are items left, sleep 
            time.sleep(random.randint(60, 120))
        else:
            # If empty, log and exit loop immediately
            log("No more keywords pending. Stopping immediately.")
            break

if __name__ == "__main__":
    main()