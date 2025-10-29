import os, io, sys, json, re
import pandas as pd

CSV_PATH = "datafpend.csv"          # <-- change if needed
OUTPUT_JSON = "fpend_data.json"

REQUIRED_COLS = ["Domain", "Subdomain", "Sumber", "Soalan", "Jawapan"]

def normalize_colname(c):
    if c is None: return ""
    # Remove BOM, trim, collapse spaces, title-case known keys
    c = str(c).replace("\ufeff", "").strip()
    c = re.sub(r"\s+", " ", c)
    # Common variants mapping (case/spacing-insensitive)
    m = c.lower()
    mapping = {
        "domain": "Domain",
        "subdomain": "Subdomain",
        "sub-domain": "Subdomain",
        "sumber": "Sumber",
        "soalan": "Soalan",
        "jawapan": "Jawapan",
    }
    return mapping.get(m, c)

def ensure_required_columns(df):
    # Normalize existing columns
    df.columns = [normalize_colname(c) for c in df.columns]
    # Try to find columns even if slightly off (e.g., trailing spaces)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}\n"
                         f"Found columns: {list(df.columns)}")
    return df

def try_read_as_csv(path):
    # Try raw bytes -> decode with multiple encodings -> pandas via StringIO with auto-sniff sep
    encodings = ["utf-8-sig", "utf-8", "cp1252", "iso-8859-1", "windows-1254"]
    for enc in encodings:
        try:
            with open(path, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode(enc)                 # strict
            except UnicodeDecodeError:
                text = raw.decode(enc, errors="replace")  # keep going with replacement
            # Let pandas sniff the delimiter
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            print(f"✅ Loaded as CSV using encoding='{enc}' (sep auto-detected).")
            return df
        except Exception as e:
            print(f"❌ CSV read failed with encoding='{enc}': {e}")
    return None

def try_read_as_excel(path):
    # If user accidentally saved as xlsx/xls or renamed
    try:
        df = pd.read_excel(path, engine="openpyxl")
        print("✅ Loaded as Excel (openpyxl).")
        return df
    except Exception as e:
        print(f"❌ Excel read failed: {e}")
        return None

def build_records(df):
    # Forward-fill hierarchical fields
    for col in ["Domain", "Subdomain", "Sumber"]:
        df[col] = df[col].ffill()

    # Drop rows without question
    df = df.dropna(subset=["Soalan"]).reset_index(drop=True)

    # Clean text
    for col in REQUIRED_COLS:
        df[col] = df[col].astype(str).str.replace("\ufeff","").str.strip()

    # Numbering per (Sumber, Subdomain)
    df["qnum"] = df.groupby(["Sumber", "Subdomain"]).cumcount() + 1
    df["id"] = "[" + df["Sumber"] + "+" + df["Subdomain"] + "+Q" + df["qnum"].astype(str) + "]"

    # Records
    records = []
    for _, r in df.iterrows():
        records.append({
            "id": r["id"],
            "document": f"Soalan: {r['Soalan']}\n\nJawapan: {r['Jawapan']}",
            "metadata": {
                "Domain": r["Domain"],
                "Subdomain": r["Subdomain"],
                "Sumber": r["Sumber"],
                "Fakulti": "Pendidikan",
            }
        })
    return records

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        sys.exit(1)

    df = None

    # 1) Try CSV path as actual CSV (robust)
    df = try_read_as_csv(CSV_PATH)

    # 2) If that fails, try reading as Excel (in case it’s actually .xlsx)
    if df is None:
        df = try_read_as_excel(CSV_PATH)

    if df is None:
        print("❌ Unable to parse the file as CSV or Excel.")
        print("👉 Open the file in Excel and 'Save As' → 'CSV UTF-8 (Comma delimited) (*.csv)'.")
        sys.exit(1)

    # Normalize/validate columns
    df = ensure_required_columns(df)

    # Build records and write JSON
    records = build_records(df)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON created: {OUTPUT_JSON}")
    print(f"Total records: {len(records)}")

if __name__ == "__main__":
    main()
