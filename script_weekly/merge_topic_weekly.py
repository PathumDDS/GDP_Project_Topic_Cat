import os
import pandas as pd

# ----------------- Configuration -----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KW_DIR = os.path.join(ROOT, "keywords_weekly")
RAW_WEEKLY_DIR = os.path.join(ROOT, "data_weekly", "raw_weekly")
OUTPUT_DIR = os.path.join(ROOT, "data_weekly", "final_dataset")
PROCESSED_FILE = os.path.join(KW_DIR, "processed.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Helper -----------------
# We need this to convert "social security" in the text file 
# to "social_security" (the filename)
def sanitize_for_filename(name):
    s = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)
    s = s.strip().replace(" ", "_")
    return s if s else "keyword"

def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        # distinct set to avoid duplicates if keyword appears twice
        return list(dict.fromkeys([l.rstrip("\n") for l in f if l.strip()]))

# ----------------- Main -----------------
def main():
    print("--- Starting Controlled Merge (via processed.txt) ---")
    
    # 1. Read the Approved List
    if not os.path.exists(PROCESSED_FILE):
        print(f"❌ Error: processed.txt not found at {PROCESSED_FILE}")
        return

    approved_keywords = read_lines(PROCESSED_FILE)
    
    if not approved_keywords:
        print("processed.txt is empty. Nothing to merge.")
        return

    print(f"Found {len(approved_keywords)} keywords in processed.txt.")

    # 2. Collect Dataframes
    df_list = []
    missing_files = []

    for kw in approved_keywords:
        # Convert keyword to expected filename
        safe_kw = sanitize_for_filename(kw)
        filename = f"{safe_kw}_weekly.csv"
        file_path = os.path.join(RAW_WEEKLY_DIR, filename)

        if os.path.exists(file_path):
            try:
                # Read CSV
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Use the sanitized keyword as column name
                if len(df.columns) > 0:
                    df.columns = [safe_kw]
                    df_list.append(df)
                    print(f"Loaded: {kw} -> {filename}")
                else:
                    print(f"⚠️ Warning: File empty for {kw}")

            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        else:
            # Log missing files (in case processed.txt has a keyword but file is deleted)
            missing_files.append(kw)

    # 3. Report Missing Files
    if missing_files:
        print("\n⚠️ WARNING: The following keywords are in processed.txt but have NO CSV file:")
        for mk in missing_files:
            print(f"   - {mk}")
        print("-" * 30)

    # 4. Merge and Save
    if df_list:
        print(f"\nMerging {len(df_list)} files...")
        final_df = pd.concat(df_list, axis=1, join='outer')
        final_df = final_df.sort_index()

        output_path = os.path.join(OUTPUT_DIR, "weekly_merged_data_.csv")
        final_df.to_csv(output_path)
        
        print("-" * 30)
        print(f"✅ Success! Master Dataset Created.")
        print(f"Location: {output_path}")
        print(f"Dimensions: {final_df.shape} (Rows=Weeks, Cols=Variables)")
        print("-" * 30)
    else:
        print("No valid data found to merge.")

if __name__ == "__main__":
    main()