import os, time, random
import pandas as pd
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timedelta, timezone
from pytrends.request import TrendReq

# --- USER CONFIGURATION (UPDATE THIS) ---
# The Start Date should be consistent for "Update Window" (e.g. 2023-01-01)

UPDATE_START_DATE = "2023-01-01" 
UPDATE_END_DATE   = "2026-01-30" 

# ----------------- Paths -----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KW_DIR = os.path.join(ROOT, "keywords_weekly")
RAW_WINDOWS = os.path.join(ROOT, "data_weekly", "raw_windows")
RAW_WEEKLY = os.path.join(ROOT, "data_weekly", "raw_weekly")
LOGS = os.path.join(ROOT, "logs_weekly")
UNPRO = os.path.join(KW_DIR, "unprocessed.txt")
PROCING = os.path.join(KW_DIR, "processing.txt")
PROCED = os.path.join(KW_DIR, "processed.txt")
FAILED = os.path.join(KW_DIR, "failed.txt")
RUN_LOG = os.path.join(LOGS, "runs.log")

GEO = "LK"
TZ = 330
MAX_RETRIES = 3

# ----------------- Helpers -----------------
def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} - {msg}\n")

def read_lines(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def pop_keyword():
    lines = read_lines(UNPRO)
    if not lines: return None
    first = lines[0]
    rest = lines[1:]
    with open(UNPRO, "w", encoding="utf-8") as f:
        for r in rest: f.write(r + "\n")
    return first

def sanitize_for_filename(name):
    s = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)
    return s.strip().replace(" ", "_")

def find_category_id(cat_tree, target_name):
    if cat_tree['name'].lower() == target_name.lower(): return cat_tree['id']
    if 'children' in cat_tree:
        for child in cat_tree['children']:
            res = find_category_id(child, target_name)
            if res: return res
    return None

def save_status(keyword, target_file):
    if os.path.exists(PROCING):
        lines = read_lines(PROCING)
        lines = [l for l in lines if l != keyword]
        with open(PROCING, "w", encoding="utf-8") as f:
            for l in lines: f.write(l + "\n")
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(keyword + "\n")

# ----------------- 1. Fetch Logic -----------------
def fetch_window(pytrends, cat_id, start_str, end_str, name):
    s_date = datetime.strptime(start_str, "%Y-%m-%d")
    e_date = datetime.strptime(end_str, "%Y-%m-%d")
    
    # Adjust to Sunday-Sunday weeks
    s_adj = s_date - timedelta(days=(s_date.weekday() + 1) % 7)
    e_adj = e_date + timedelta(days=(6 - e_date.weekday()) % 7)
    
    timeframe = f"{s_adj:%Y-%m-%d} {e_adj:%Y-%m-%d}"
    full_idx = pd.date_range(s_adj, e_adj, freq="W-SUN")
    
    log(f"   Requesting: {timeframe}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1: time.sleep(random.randint(30, 60))
            
            pytrends.build_payload(kw_list=[""], timeframe=timeframe, geo=GEO, cat=int(cat_id))
            df = pytrends.interest_over_time()

            if df is not None and not df.empty:
                if "isPartial" in df.columns: df = df.drop(columns=["isPartial"])
                if "" in df.columns: df = df.rename(columns={"": name})
                elif "all" in df.columns: df = df.rename(columns={"all": name})
                if name not in df.columns and len(df.columns) > 0: df.columns = [name]
                
                if df[name].sum() == 0:
                    return pd.DataFrame(index=full_idx, columns=[name]).fillna(0)
                return df.reindex(full_idx).fillna(0)
            
            return pd.DataFrame(index=full_idx, columns=[name]).fillna(0)
        except Exception as e:
            log(f"   Error (Attempt {attempt}): {e}")
            time.sleep(60)

    return pd.DataFrame(index=full_idx, columns=[name]).fillna(0)

# ----------------- 2. Filter Redundant Files (THE FIX) -----------------
def get_best_files(raw_win_folder):
    """
    Scans the folder and REMOVES subsets.
    If we have [20230101_20230331] and [20230101_20231231], 
    it keeps ONLY [20230101_20231231].
    """
    files = [f for f in os.listdir(raw_win_folder) if f.endswith(".csv")]
    if not files: return []

    # 1. Parse Metadata: (Start_Date, End_Date, Filename)
    file_meta = []
    for f in files:
        try:
            # Expected format: Name_START_END.csv
            parts = f.replace(".csv", "").split("_")
            s_str = parts[-2]
            e_str = parts[-1]
            s_date = datetime.strptime(s_str, "%Y%m%d")
            e_date = datetime.strptime(e_str, "%Y%m%d")
            file_meta.append({'file': f, 'start': s_date, 'end': e_date})
        except:
            continue # Skip files that don't match pattern

    # 2. Filter Logic
    # We keep a file ONLY if no other file completely contains it.
    keep_files = []
    for candidate in file_meta:
        is_redundant = False
        for other in file_meta:
            if candidate['file'] == other['file']: continue
            
            # Check if 'other' covers 'candidate' completely
            # i.e. Other starts before/at same time AND ends later
            if (other['start'] <= candidate['start']) and (other['end'] > candidate['end']):
                is_redundant = True
                break
        
        if not is_redundant:
            keep_files.append(candidate)

    # 3. Sort by Start Date (Required for stitching)
    keep_files.sort(key=lambda x: x['start'])
    
    final_list = [x['file'] for x in keep_files]
    log(f"   Filtering: Found {len(files)} files -> Keeping {len(final_list)} unique windows.")
    return final_list

# ----------------- 3. Stitch Logic -----------------
def stitch_files(raw_win_folder, file_list):
    if not file_list: return None
    
    # Load DFs
    dfs = []
    for f in file_list:
        path = os.path.join(raw_win_folder, f)
        dfs.append(pd.read_csv(path, index_col=0, parse_dates=True))
    
    # Stitch
    stitched = dfs[0].copy().sort_index()
    
    for i in range(1, len(dfs)):
        df_new = dfs[i]
        
        # Overlap
        overlap_start = max(stitched.index[0], df_new.index[0])
        overlap_end   = min(stitched.index[-1], df_new.index[-1])
        
        slice_old = stitched.loc[overlap_start:overlap_end]
        slice_new = df_new.loc[overlap_start:overlap_end]
        
        if slice_old.empty or slice_new.empty:
            scale = 1.0
        else:
            m_old = slice_old.mean().iloc[0]
            m_new = slice_new.mean().iloc[0]
            scale = m_old / m_new if m_new > 0 and m_old > 0 else 1.0

        # Append Scaled Data (Cut off overlap from new)
        last_stitch_date = stitched.index[-1]
        df_new_scaled = df_new * scale
        fresh_data = df_new_scaled[df_new_scaled.index > last_stitch_date]
        
        stitched = pd.concat([stitched, fresh_data])

    return stitched[~stitched.index.duplicated(keep="last")].sort_index()

# ----------------- Main -----------------
def main():
    while True:
        cat_name = pop_keyword()
        if not cat_name: break
        
        with open(PROCING, "a", encoding="utf-8") as f: f.write(cat_name + "\n")
        safe_name = sanitize_for_filename(cat_name)
        log(f"--- PROCESSING: {cat_name} ---")

        pytrends = TrendReq(hl="en-US", tz=TZ, requests_args={'verify': False})

        # 1. Resolve ID
        try:
            full_tree = pytrends.categories()
            cat_id = find_category_id(full_tree, cat_name)
            if not cat_id:
                log(f"   FAIL: ID not found")
                save_status(cat_name, FAILED)
                continue
        except:
            save_status(cat_name, FAILED)
            continue

        # 2. Fetch NEW Window
        new_window_df = fetch_window(pytrends, cat_id, UPDATE_START_DATE, UPDATE_END_DATE, cat_name)
        
        # 3. Save NEW Window
        win_dir = os.path.join(RAW_WINDOWS, safe_name)
        os.makedirs(win_dir, exist_ok=True)
        
        s_str = datetime.strptime(UPDATE_START_DATE, "%Y-%m-%d").strftime("%Y%m%d")
        e_str = datetime.strptime(UPDATE_END_DATE,   "%Y-%m-%d").strftime("%Y%m%d")
        fname = f"{safe_name}_{s_str}_{e_str}.csv"
        new_window_df.to_csv(os.path.join(win_dir, fname))
        log(f"   Saved: {fname}")

        # 4. Filter & Stitch
        # This function scans the folder, ignores old subsets, and keeps only the latest windows
        best_files = get_best_files(win_dir)
        final_df = stitch_files(win_dir, best_files)
        
        if final_df is not None:
            final_path = os.path.join(RAW_WEEKLY, f"{safe_name}_weekly.csv")
            final_df.to_csv(final_path)
            log(f"   SUCCESS: Stitched {len(best_files)} windows -> {len(final_df)} weeks.")
            save_status(cat_name, PROCED)
        else:
            save_status(cat_name, FAILED)

        # Sleep
        if read_lines(UNPRO): time.sleep(random.randint(45, 90))
        else: break

if __name__ == "__main__":
    main()