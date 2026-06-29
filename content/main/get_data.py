import fastf1
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
CACHE_DIR = './cache'   # Relative path for local storage
fastf1.Cache.enable_cache(CACHE_DIR)

def load_race_session(year, circuit):
    """
    Loads the Race session (Lightweight - mainly for timing/strategy context).
    """
    print(f"--> [NETWORK] Loading Race Session: {year} {circuit}...")
    session = fastf1.get_session(year, circuit, 'R')
    # We keep telemetry=False for the race to save memory, 
    # unless you want to analyze specific race laps later.
    session.load(telemetry=False, weather=True) 
    return session

def get_style_data(year, circuit, driver_code):
    """
    Loads the Qualifying session to extract the 'Golden Lap' telemetry.
    This serves as the 'DNA' for the driving style (Throttle/Brake maps).
    """
    print(f"--> [NETWORK] Loading Quali Session for Style Analysis...")
    session = fastf1.get_session(year, circuit, 'Q')
    
    # We NEED telemetry here to get throttle/brake traces
    session.load(telemetry=True, weather=False) 
    
    print(f"--> [INGEST] Extracting Fastest Lap Telemetry for {driver_code}...")
    try:
        laps = session.laps.pick_drivers(driver_code)
        fastest_lap = laps.pick_fastest()
        
        # Get the detailed physics data (Speed, Throttle, Brake, Gear, RPM, etc.)
        # adding 'Distance' is crucial for mapping it to the track
        telemetry = fastest_lap.get_telemetry()
        
        # Metadata for normalization
        stats = {
            'LapTime': fastest_lap['LapTime'].total_seconds(),
            'Compound': fastest_lap['Compound'],
            'TyreLife': fastest_lap['TyreLife']
        }
        
        return telemetry, stats
    
    except Exception as e:
        print(f"!! [ERROR] Could not extract style data: {e}")
        return None, None

def get_driver_race_data(session, driver_code):
    """
    Extracts race timing data for a specific driver (Hero or Rival).
    """
    print(f"--> [INGEST] Extracting Race Data for {driver_code}...")
    laps = session.laps.pick_drivers(driver_code)
    
    cols = [
        'LapNumber', 'Stint', 'Compound', 'TyreLife', 
        'LapTime', 'Time', 'PitInTime', 'PitOutTime', 
        'TrackStatus' # Useful to filter out Safety Car laps later
    ]
    
    df = laps[cols].copy()
    
    # Cleaning
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    df['RaceTimeSeconds'] = df['Time'].dt.total_seconds()
    
    return df

def get_weather_data(session):
    """
    Extracts environmental variables.
    """
    print(f"--> [INGEST] Extracting Weather Conditions...")
    weather_df = session.weather_data
    cols = ['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
    return weather_df[cols]

# --- ORCHESTRATOR ---
def run_ingestion():
    # 1. Inputs
    print("\n--- CONFIGURATION ---")
    YEAR = int(input("Enter Year (e.g. 2024): "))
    CIRCUIT = input("Enter Circuit (e.g. Monaco): ")
    HERO_CODE = input("Enter Hero Driver (e.g. PIA): ")
    RIVAL_CODE = input("Enter Rival Driver (e.g. VER): ")
    
    # 2. Get The "Style" (High Quality Telemetry from Q3)
    style_telemetry, style_stats = get_style_data(YEAR, CIRCUIT, HERO_CODE)
    
    # 3. Get The "Race Context" (Laps, Gaps, Weather)
    race_session = load_race_session(YEAR, CIRCUIT)
    
    hero_race_df = get_driver_race_data(race_session, HERO_CODE)
    rival_race_df = get_driver_race_data(race_session, RIVAL_CODE)
    weather_df = get_weather_data(race_session)
    
    print(f"\n--- INGESTION COMPLETE ---")
    print(f"Style Data: {len(style_telemetry)} points (Ref Time: {style_stats['LapTime']}s)")
    print(f"Hero Race Laps: {len(hero_race_df)}")
    print(f"Rival Race Laps: {len(rival_race_df)}")
    
    # 4. Packaging
    # We return a dictionary so the next file is clean
    data_bundle = {
        'metadata': {'year': YEAR, 'circuit': CIRCUIT, 'hero': HERO_CODE, 'rival': RIVAL_CODE},
        'style': {
            'telemetry': style_telemetry, # Pandas DF with Speed, Throttle, Brake
            'stats': style_stats          # Metadata about that push lap
        },
        'race': {
            'hero_laps': hero_race_df,
            'rival_laps': rival_race_df,
            'weather': weather_df
        }
    }
    
    return data_bundle

# --- SELF-TEST ---
if __name__ == "__main__":
    data = run_ingestion()
    
    # Visual Sanity Check
    print("\n[Preview: Style Telemetry]")
    print(data['style']['telemetry'][['Distance', 'Speed', 'Throttle', 'Brake']].head())