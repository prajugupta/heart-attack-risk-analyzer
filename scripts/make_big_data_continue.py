from pathlib import Path
import zipfile
import requests
import numpy as np
import pandas as pd
import pyreadstat


RAW_DIR = Path("data/raw_multi")
CLEAN_DIR = Path("data/clean")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = CLEAN_DIR / "heart_big_data.csv"

# Continue only failed/pending years
YEARS = [2022, 2023, 2024]

TARGET_ALIASES = ["CVDINFR4"]

FEATURE_ALIASES = {
    "state": ["_STATE"],
    "age_group": ["_AGEG5YR"],
    "sex": ["_SEX"],

    "general_health": ["GENHLTH"],
    "physical_bad_days": ["PHYSHLTH"],
    "mental_bad_days": ["MENTHLTH"],

    "bmi": ["_BMI5"],

    "physical_activity": ["_TOTINDA"],
    "smoking_status": ["_SMOKER3"],
    "heavy_drinking": ["_RFDRHV9"],
    "sleep_hours": ["SLEPTIM1"],

    "diabetes": ["DIABETE4", "DIABETE3"],
    "stroke": ["CVDSTRK3"],
    "kidney_disease": ["CHCKDNY2", "CHCKDNY1"],
    "asthma": ["ASTHMA3"],
    "copd": ["CHCCOPD3"],
    "depression": ["ADDEPEV3"],
    "arthritis": ["HAVARTH4", "HAVARTH3"],

    "high_bp": ["BPHIGH6", "BPHIGH5", "BPHIGH4"],
    "high_cholesterol": ["TOLDHI3", "TOLDHI2"],

    "insurance": ["PRIMINS2", "HLTHPLN1"],
    "personal_doctor": ["PERSDOC3", "PERSDOC2"],
    "cost_barrier": ["MEDCOST1"],
    "checkup": ["CHECKUP1"],
}


def read_xport_safe(xpt_path, usecols=None, metadataonly=False):
    encodings = [None, "latin1", "cp1252", "ISO-8859-1"]

    last_error = None

    for enc in encodings:
        try:
            if metadataonly:
                return pyreadstat.read_xport(
                    str(xpt_path),
                    metadataonly=True,
                    encoding=enc
                )
            else:
                return pyreadstat.read_xport(
                    str(xpt_path),
                    usecols=usecols,
                    encoding=enc
                )
        except Exception as e:
            last_error = e
            print(f"Read failed with encoding={enc}: {e}")

    raise last_error


def download_year(year: int) -> Path | None:
    url = f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"
    zip_path = RAW_DIR / f"LLCP{year}XPT.zip"
    year_dir = RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"\nDownloading {year} data...")
        try:
            r = requests.get(url, timeout=180)
            if r.status_code != 200:
                print(f"Download failed for {year}. Status: {r.status_code}")
                return None

            zip_path.write_bytes(r.content)
            print(f"Downloaded: {zip_path}")
        except Exception as e:
            print(f"Download error for {year}: {e}")
            return None
    else:
        print(f"\nZIP already exists for {year}")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(year_dir)
    except Exception as e:
        print(f"Extraction error for {year}: {e}")
        return None

    xpt_files = list(year_dir.glob("*.XPT")) + list(year_dir.glob("*.xpt"))

    if not xpt_files:
        print(f"No XPT file found for {year}")
        return None

    return xpt_files[0]


def pick_available_columns(xpt_path: Path):
    _, meta = read_xport_safe(xpt_path, metadataonly=True)
    available = set(meta.column_names)

    selected_raw_cols = []
    rename_map = {}

    target_raw = None

    for t in TARGET_ALIASES:
        if t in available:
            target_raw = t
            selected_raw_cols.append(t)
            rename_map[t] = "heart_attack_raw"
            break

    if target_raw is None:
        return None, None

    for clean_name, aliases in FEATURE_ALIASES.items():
        found = None

        for alias in aliases:
            if alias in available:
                found = alias
                break

        if found is not None:
            selected_raw_cols.append(found)
            rename_map[found] = clean_name

    selected_raw_cols = list(dict.fromkeys(selected_raw_cols))

    return selected_raw_cols, rename_map


def clean_year_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()

    df = df[df["heart_attack_raw"].isin([1, 2])].copy()
    df["heart_attack"] = df["heart_attack_raw"].map({1: 1, 2: 0})
    df = df.drop(columns=["heart_attack_raw"])

    df["year"] = year

    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].replace([9999, 99999], np.nan)
        df["bmi"] = df["bmi"] / 100

    for col in ["physical_bad_days", "mental_bad_days"]:
        if col in df.columns:
            df[col] = df[col].replace(88, 0)

    missing_codes = [7, 9, 77, 99, 777, 999, 7777, 9999]

    for col in df.columns:
        if col in ["heart_attack", "year", "state"]:
            continue
        df[col] = df[col].replace(missing_codes, np.nan)

    if "sleep_hours" in df.columns:
        df.loc[(df["sleep_hours"] < 1) | (df["sleep_hours"] > 24), "sleep_hours"] = np.nan

    required_cols = list(FEATURE_ALIASES.keys()) + ["year", "heart_attack"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required_cols]

    return df


def main():
    if not OUTPUT_FILE.exists():
        print("Main big data file not found.")
        print("Expected:", OUTPUT_FILE)
        print("Run make_big_data.py first or change this script to start from 2014.")
        return

    existing_rows = sum(1 for _ in open(OUTPUT_FILE, "r", encoding="utf-8", errors="ignore")) - 1
    print("Existing rows in big data:", f"{existing_rows:,}")

    total_added = 0

    for year in YEARS:
        xpt_path = download_year(year)

        if xpt_path is None:
            print(f"Skipping {year}")
            continue

        selected_cols, rename_map = pick_available_columns(xpt_path)

        if selected_cols is None:
            print(f"Target not found for {year}. Skipping.")
            continue

        print(f"Loading selected columns for {year}...")
        df, _ = read_xport_safe(xpt_path, usecols=selected_cols)

        df = df.rename(columns=rename_map)
        cleaned = clean_year_data(df, year)

        for col in cleaned.columns:
            if col == "heart_attack":
                continue
            if cleaned[col].isna().sum() > 0:
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

        cleaned.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)

        total_added += len(cleaned)

        print(f"Year {year} done. Rows added: {len(cleaned):,}")
        print(f"Total added in this run: {total_added:,}")

    final_rows = sum(1 for _ in open(OUTPUT_FILE, "r", encoding="utf-8", errors="ignore")) - 1

    print("\nContinue process completed.")
    print("Rows added:", f"{total_added:,}")
    print("Final total rows:", f"{final_rows:,}")
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()