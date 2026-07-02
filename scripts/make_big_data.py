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

YEARS = list(range(2014, 2025))

# Main target: heart attack / myocardial infarction
TARGET_ALIASES = ["CVDINFR4"]

# Canonical feature name -> possible BRFSS raw variable names
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
    _, meta = pyreadstat.read_xport(str(xpt_path), metadataonly=True)
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

    # Keep only valid heart attack target
    df = df[df["heart_attack_raw"].isin([1, 2])].copy()
    df["heart_attack"] = df["heart_attack_raw"].map({1: 1, 2: 0})
    df = df.drop(columns=["heart_attack_raw"])

    df["year"] = year

    # BMI conversion before general missing cleaning
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].replace([9999, 99999], np.nan)
        df["bmi"] = df["bmi"] / 100

    # 88 means zero bad health days in these columns
    for col in ["physical_bad_days", "mental_bad_days"]:
        if col in df.columns:
            df[col] = df[col].replace(88, 0)

    # Missing/non-response codes
    missing_codes = [7, 9, 77, 99, 777, 999, 7777, 9999]

    for col in df.columns:
        if col in ["heart_attack", "year", "state"]:
            continue
        df[col] = df[col].replace(missing_codes, np.nan)

    # Sleep sanity
    if "sleep_hours" in df.columns:
        df.loc[(df["sleep_hours"] < 1) | (df["sleep_hours"] > 24), "sleep_hours"] = np.nan

    # Ensure all canonical columns exist
    required_cols = list(FEATURE_ALIASES.keys()) + ["year", "heart_attack"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required_cols]

    return df


def main():
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    used_years = []

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
        df, _ = pyreadstat.read_xport(str(xpt_path), usecols=selected_cols)
        df = df.rename(columns=rename_map)

        cleaned = clean_year_data(df, year)

        # Median fill year-wise
        for col in cleaned.columns:
            if col == "heart_attack":
                continue
            if cleaned[col].isna().sum() > 0:
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

        write_header = not OUTPUT_FILE.exists()
        cleaned.to_csv(OUTPUT_FILE, mode="a", index=False, header=write_header)

        total_rows += len(cleaned)
        used_years.append(year)

        print(f"Year {year} done. Rows kept: {len(cleaned):,}")
        print(f"Total rows so far: {total_rows:,}")

    print("\nBig dataset completed.")
    print("Used years:", used_years)
    print("Total rows:", f"{total_rows:,}")
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()