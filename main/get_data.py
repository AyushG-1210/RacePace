import fastf1
import pandas as pd
import tabulate as tb

# 1. Setup Caching (So we don't re-download 50MB every time)
fastf1.Cache.enable_cache('../cache') 

def get_race_data(year, circuit, driver_code):
    print(f"Loading {year} {circuit} for {driver_code}...")
    
    # Load the Race Session
    session = fastf1.get_session(year, circuit, 'Race')
    session.load(telemetry=False, weather=False) # Keep it light for now
    
    # Pick the Driver
    laps = session.laps.pick_drivers(driver_code)
    
    # Select only what we need for the Model
    # LapTime is a "Timedelta" object (e.g., "0 days 00:01:15.023")
    # We convert it to seconds (float) for math.
    clean_laps = laps[['LapNumber', 'LapTime', 'TyreLife', 'Compound']].copy()
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
    
    # Drop rows with no time (Pit stops or errors)
    clean_laps = clean_laps.dropna(subset=['LapTimeSeconds'])
    
    return clean_laps

# Test it immediately
if __name__ == "__main__":
    df = get_race_data(2025, 'Zandvoort', 'PIA')
    print("\n--- DATA SUCCESSFULLY LOADED ---")
    print(tb.tabulate(df.tail(10), headers='keys', tablefmt='rounded_grid', numalign="center", stralign="center")) # Show last 10 laps
