import fastf1
import pandas as pd
import numpy as np
import subprocess

# --- CONFIGURATION ---
CACHE_DIR = './cache'   # Relative path for local storage
fastf1.Cache.enable_cache(CACHE_DIR)
subprocess.run("rm -rf cache/*", shell=True, capture_output=True)

circuit = input("Enter Circuit Code (e.g., 'Zandvoort'): ")
year = int(input("Enter Year (e.g., 2025): "))
age = int(input("Enter Current Tire Age (in laps, e.g., 5): "))

def load_session(year=year, circuit=circuit, session_type='R'):
    """
    Loads the race session object. This is the heavy network call.
    """
    print(f"--> [NETWORK] Loading Session: {year} {circuit} ({session_type})...")
    session = fastf1.get_session(year, circuit, session_type)
    session.load(telemetry=False, weather=True) # Weather=True is crucial for physics
    return session, age 

def get_hero_data(session, driver_code='PIA'):
    """
    Extracts high-fidelity data for the Hero Driver (Piastri).
    """
    print(f"--> [INGEST] Extracting Hero Data for {driver_code}...")
    
    # 1. Pick the driver
    laps = session.laps.pick_drivers(driver_code)
    
    # 2. Select columns relevant for Physics/PINN
    # We need: Time (for order), LapTime (target), TireLife (input), Compound (context)
    cols = [
        'LapNumber', 'Stint', 'Compound', 'TireLife', 
        'LapTime', 'Time', 'PitInTime', 'PitOutTime'
    ]
    
    # Create a copy to avoid SettingWithCopy warnings later
    hero_df = laps[cols].copy()
    
    # 3. Convert Timedeltas to Seconds (Float) for Math
    hero_df['LapTimeSeconds'] = hero_df['LapTime'].dt.total_seconds()
    hero_df['RaceTimeSeconds'] = hero_df['Time'].dt.total_seconds()
    
    return hero_df

def get_global_data(session):
    """
    Extracts 'Ghost Data' for the rest of the grid (Traffic Map).
    """
    print(f"--> [INGEST] Extracting Global Grid Data...")
    
    # We just need to know WHERE everyone is.
    # Driver, LapNumber, and the cumulative RaceTime (when they crossed the line)
    global_df = session.laps[['Driver', 'LapNumber', 'Time', 'LapTime']].copy()
    
    # Convert to seconds
    global_df['RaceTimeSeconds'] = global_df['Time'].dt.total_seconds()
    global_df['LapTimeSeconds'] = global_df['LapTime'].dt.total_seconds()
    
    return global_df

def get_weather_data(session):
    """
    Extracts environmental variables.
    """
    print(f"--> [INGEST] Extracting Weather Conditions...")
    
    # FastF1 returns weather minute-by-minute. 
    # For Phase 1, we'll return the full dataframe so Step 2 can align it with laps.
    weather_df = session.weather_data
    
    # Keep only what impacts tires
    cols = ['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
    return weather_df[cols]

# --- ORCHESTRATOR ---
def run_ingestion():
    # Hardcoded Inputs as requested
    YEAR = 2025
    CIRCUIT = 'Zandvoort'
    DRIVER = 'PIA'
    
    # 1. Load Session
    session = load_session(YEAR, CIRCUIT)
    
    # 2. Extract Components
    hero_df = get_hero_data(session, DRIVER)
    global_df = get_global_data(session)
    weather_df = get_weather_data(session)
    
    print(f"\n--- INGESTION COMPLETE ---")
    print(f"Hero Laps (PIA): {len(hero_df)} rows")
    print(f"Global Laps: {len(global_df)} rows")
    print(f"Weather Records: {len(weather_df)} rows")
    
    return hero_df, global_df, weather_df, 

# --- SELF-TEST ---
if __name__ == "__main__":
    h, g, w = run_ingestion()
    
    # Visual Check
    print("\n[Preview: Hero Data]")
    print(h[['LapNumber', 'LapTimeSeconds', 'TireLife']].head())