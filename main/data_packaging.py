import pandas as pd
import numpy as np

def identify_stints(df):
    """
    Assigns a unique ID to each stint.
    Logic: New Stint = Change in Compound OR Reset in TyreLife.
    """
    print("--> [PACKAGE] Identifying Stints...")
    
    # 1. Did the Tyre age reset? (e.g., 20 -> 1)
    # We use .shift(1) to compare current row with previous row
    df['Tyre_Reset'] = df['TyreLife'] < df['TyreLife'].shift(1)
    
    # 2. Did the compound change? (e.g., SOFT -> HARD)
    df['Compound_Change'] = df['Compound'] != df['Compound'].shift(1)
    
    # 3. Cumulative Sum gives us the ID (0, 1, 2...)
    # We fillna(0) for the very first lap
    df['Stint_ID'] = (df['Tyre_Reset'] | df['Compound_Change']).cumsum().fillna(0).astype(int)
    
    return df

def merge_weather_data(laps_df, weather_df):
    """
    Matches the closest weather recording to each lap time.
    """
    print("--> [PACKAGE] Merging Weather Data...")
    
    # ensure both Time columns are Timedeltas for sorting
    laps_df = laps_df.sort_values('Time')
    weather_df = weather_df.sort_values('Time')
    
    # Use merge_asof: It finds the LAST weather entry before the lap finished
    # direction='backward' means "What was the temp when I crossed the line?"
    merged_df = pd.merge_asof(
        laps_df, 
        weather_df[['Time', 'TrackTemp', 'AirTemp']], 
        on='Time', 
        direction='backward'
    )
    
    return merged_df

def add_fuel_correction(df, initial_fuel_kg=110, time_loss_per_kg=0.035):
    """
    Calculates the 'True' Tyre Pace by removing the fuel weight advantage.
    """
    print("--> [PACKAGE] Applying Physics (Fuel Correction)...")
    
    # Standard assumption: Fuel burns linearly from 110kg to 0kg over 72 laps
    total_laps = 72
    fuel_burn_per_lap = initial_fuel_kg / total_laps
    
    # Calculate Fuel Burned at specific lap
    df['Fuel_Kg_Burned'] = (df['LapNumber'] - 1) * fuel_burn_per_lap
    
    # The time we need to ADD back. 
    # (The car is faster than it should be because it's light. We punish it.)
    fuel_penalty = df['Fuel_Kg_Burned'] * time_loss_per_kg
    
    df['FuelCorrectedTime'] = df['LapTimeSeconds'] + fuel_penalty
    return df

def filter_warmup_laps(df):
    """
    Removes the first lap of every stint (The 'Settle-in' lap).
    """
    print("--> [PACKAGE] Removing Warmup Laps...")
    
    # Group by Stint and remove the entry with the lowest TyreLife (the start)
    # Alternatively, just drop the first row of the group.
    
    # Logic: Keep rows where TyreLife is NOT the minimum of that stint
    # (This handles cases where you might enter track on old Tyres)
    
    clean_stints = []
    for stint_id, group in df.groupby('Stint_ID'):
        if len(group) > 1: # Only drop if we have enough data
            # Drop the first chronological lap of the stint
            valid_laps = group.iloc[1:].copy() 
            clean_stints.append(valid_laps)
            
    if not clean_stints:
        return df # Return empty if everything was filtered
        
    return pd.concat(clean_stints)

# --- MASTER FUNCTION ---
def process_packaging(clean_laps, weather_data):
    
    # 1. Identify Stints
    df = identify_stints(clean_laps)
    
    # 2. Merge Weather
    df = merge_weather_data(df, weather_data)
    
    # 3. Apply Fuel Correction (The Physics Target)
    df = add_fuel_correction(df)
    
    # 4. Remove Warmup Laps
    final_df = filter_warmup_laps(df)
    
    # 5. Final Selection of Columns for PINN
    cols = ['Stint_ID', 'LapNumber', 'TyreLife', 'Compound', 
            'TrackTemp', 'FuelCorrectedTime']
    
    return final_df[cols]

# --- SELF TEST ---
if __name__ == "__main__":
    from get_data import run_ingestion
    from process_data import clean_garbage_laps # The Step 2 script
    
    # 1. Load & Clean
    raw, _, weather, age = run_ingestion()
    clean = clean_garbage_laps(raw)
    
    # 2. Package
    pinn_ready_data = process_packaging(clean, weather)
    
    print("\n[Preview: PINN Input Data]")
    print(pinn_ready_data.head(10).to_string(index=False))
    
    # Check if we have TrackTemp
    if 'TrackTemp' in pinn_ready_data.columns:
        print("\n[Success] Weather Data Merged.")
