import pandas as pd
import numpy as np

def clean_garbage_laps(df):
    """
    Filters out non-racing laps (Pit stops, Safety Cars, Lap 1).
    Returns a clean DataFrame ready for Physics analysis.
    """
    print(f"--> [PROCESS] Cleaning Garbage Laps (Initial Count: {len(df)})")
    
    # 1. Remove Lap 1 (Standing Start)
    # The first lap is always slow due to the standing start.
    df = df[df['LapNumber'] > 1].copy()
    
    # 2. Remove Pit Stop Laps (In-Laps and Out-Laps)
    # FastF1 has specific columns: 'PitInTime' and 'PitOutTime'.
    # If these are NOT Null (NaT), the driver was in the pit lane during that lap.
    # We want rows where BOTH are Null.
    df = df[pd.isna(df['PitInTime']) & pd.isna(df['PitOutTime'])].copy()
    
    # 3. Remove Safety Car / VSC / Red Flag Laps (Statistical Filter)
    # Instead of checking complex track status flags, we use the "107% Rule".
    # If a lap is > 7% slower than the rolling average, it's not a 'push' lap.
    
    # Calculate a rolling median (robust to outliers) of the previous 5 laps
    rolling_median = df['LapTimeSeconds'].rolling(window=5, center=True).median()
    
    # Define the threshold (1.07 is a standard FIA cutoff, but we can be tighter like 1.10)
    # We fill NaN (first few laps) with the column median to prevent dropping them.
    threshold = rolling_median.fillna(df['LapTimeSeconds'].median()) * 1.07
    
    # Filter: Keep laps that are FASTER than the threshold
    clean_df = df[df['LapTimeSeconds'] < threshold].copy()
    
    print(f"--> [PROCESS] Clean Complete (Final Count: {len(clean_df)})")
    return clean_df

# --- SELF-TEST ---
if __name__ == "__main__":
    # Import the ingestor to get data to test with
    from get_data import run_ingestion
    
    # 1. Get Raw Data
    hero_raw, _, _ = run_ingestion()
    
    # 2. Clean it
    hero_clean = clean_garbage_laps(hero_raw)
    
    # 3. Verify
    print("\n[Preview: Cleaned Data]")
    print(hero_clean[['LapNumber', 'LapTimeSeconds', 'TireLife']].head(10))
    
    # Check if we accidentally deleted everything
    if len(hero_clean) == 0:
        print("WARNING: Filter removed all laps! Check thresholds.")