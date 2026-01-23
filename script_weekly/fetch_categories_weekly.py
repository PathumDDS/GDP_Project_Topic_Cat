# script_weekly/fetch_categories.py
# CATEGORY RESOLUTION VERSION
# - Resolves Category IDs from Names in unprocessed.txt
# - Maintains original window-stitching and file structure

import os, time, random, traceback
import pandas as pd
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq


# ----------------- Paths (Kept Same) -----------------
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
FIXED_END_DATE = "2025-12-31"

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
    return s if s else "category"

# ----------------- Category Search Helper -----------------
def find_category_id(cat_tree, target_name):
    """Recursively searches the Google Category tree for a name match"""
    if cat_tree['name'].lower() == target_name.lower():
        return cat_tree['id']
    if 'children' in cat_tree:
        for child in cat_tree['children']:
            result = find_category_id(child, target_name)
            if result: return result
    return None

# ----------------- Fetch Logic (Updated for Category Column Naming) -----------------
def fetch_window(pytrends, cat_id, start, end, original_name):
    start_adj = start - timedelta(days=(start.weekday() + 1) % 7) 
    end_adj = end + timedelta(days=(6 - end.weekday()) % 7)
    timeframe = f"{start_adj:%Y-%m-%d} {end_adj:%Y-%m-%d}"
    full_idx = pd.date_range(start_adj, end_adj, freq="W-SUN")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1: time.sleep(random.randint(MIN_SLEEP, MAX_SLEEP))
            
            # CATEGORY FETCH: Empty keyword list, specific category ID
            pytrends.build_payload(kw_list=[""], timeframe=timeframe, geo=GEO, cat=int(cat_id))
            df = pytrends.interest_over_time()

            if df is not None and not df.empty:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                
                # --- THE ACCURATE FIX ---
                # When keyword is [""], Google usually returns a column named "" (empty string)
                # Sometimes it returns 'all'. We check for both.
                if "" in df.columns:
                    df = df.rename(columns={"": original_name})
                elif "all" in df.columns:
                    df = df.rename(columns={"all": original_name})
                
                # If it's still not renamed (rare), rename by position (first column)
                if original_name not in df.columns and len(df.columns) > 0:
                    df.columns = [original_name]
                
                # If Google returns data but it's all zeros
                if df[original_name].sum() == 0:
                    return pd.DataFrame(index=full_idx, columns=[original_name]).fillna(0)

                return df.reindex(full_idx).fillna(0)
            
            return pd.DataFrame(index=full_idx, columns=[original_name]).fillna(0)
        except Exception as e:
            log(f"Fetch Error: {e}")
            time.sleep(60)
            
    return pd.DataFrame(index=full_idx, columns=[original_name]).fillna(0)

# ----------------- Original Stitching Logic -----------------
def stitch_windows(windows_list, name):
    if not windows_list: return None
    stitched = windows_list[0][0].copy().sort_index()
    for i in range(1, len(windows_list)):
        prev_df, prev_s, prev_e = windows_list[i - 1]
        df, s, e = windows_list[i]
        overlap_start, overlap_end = max(prev_s, s), min(prev_e, e)
        m_old = stitched.loc[overlap_start:overlap_end].mean().iloc[0]
        m_new = df.loc[overlap_start:overlap_end].mean().iloc[0]
        scale = m_old / m_new if m_new > 0 and m_old > 0 else 1.0
        stitched = pd.concat([stitched, (df * scale).loc[prev_e + timedelta(days=1):]])
    return stitched[~stitched.index.duplicated(keep="last")].sort_index()

# ----------------- Main -----------------
def main():
    while True:
        cat_name = pop_keyword()
        if not cat_name: break
        save_status_move(cat_name, PROCING)
        safe_name = sanitize_for_filename(cat_name)
        log(f"--- RESOLVING CATEGORY ID: {cat_name} ---")

        pytrends = TrendReq(hl="en-US", tz=TZ, requests_args={'verify': False})

        # --- STEP 1: RESOLVE CATEGORY NAME TO ID ---
        try:
            full_tree = pytrends.categories()
            cat_id = find_category_id(full_tree, cat_name)
            if not cat_id:
                log(f"FAIL: ID not found for Category Name: {cat_name}")
                save_status_move(cat_name, FAILED)
                continue
            log(f"Resolved {cat_name} to ID: {cat_id}")
        except Exception as e:
            log(f"Tree Error: {e}")
            save_status_move(cat_name, FAILED)
            continue

        # --- STEP 2: PROCEED WITH WINDOWED FETCH ---
        cur = START_DATE
        # Check if FIXED_END_DATE is set at the top
        if FIXED_END_DATE:
            now = datetime.strptime(FIXED_END_DATE, "%Y-%m-%d")
        else:
            # Your original auto-calculation logic
            today = datetime.now(timezone.utc).replace(tzinfo=None)
            now = (today.replace(day=1) - timedelta(days=1)) - timedelta(days=((today.replace(day=1) - timedelta(days=1)).weekday() - 6) % 7)
        
        win_list = []
        while cur + relativedelta(years=WINDOW_YEARS) <= now:
            win_list.append((cur, cur + relativedelta(years=WINDOW_YEARS)))
            cur += relativedelta(years=STEP_YEARS)
        if (now - cur).days >= 7: win_list.append((cur, now))

        collected = []
        for idx, (s, e) in enumerate(win_list):
            if idx > 0: time.sleep(random.randint(MIN_SLEEP, MAX_SLEEP))
            df = fetch_window(pytrends, cat_id, s, e, cat_name)
            collected.append((df, s, e))

        stitched = stitch_windows(collected, cat_name)
        win_dir = os.path.join(RAW_WINDOWS, safe_name)
        os.makedirs(win_dir, exist_ok=True)
        for (df, s, e) in collected:
            df.to_csv(os.path.join(win_dir, f"{safe_name}_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.csv"))
        
        stitched.to_csv(os.path.join(RAW_WEEKLY, f"{safe_name}_weekly.csv"))
        save_status_move(cat_name, PROCED)
        log(f"SUCCESS: {cat_name}")
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