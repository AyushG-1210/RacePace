import pandas as pd

def add_fuel_correction(df, initial_fuel_kg=110, time_loss_per_kg=0.035):
    """
    Adjusts lap times to remove the advantage gained by burning fuel.
    Returns the dataframe with a new column: 'FuelCorrectedTime'
    """
    total_laps = df['LapNumber'].max()
    
    # Assumption: Fuel burns linearly over the race
    fuel_burn_per_lap = initial_fuel_kg / total_laps
    
    # Calculate how much fuel is GONE at each lap
    # Lap 1: 0kg gone. Lap 72: 110kg gone.
    df['FuelBurned'] = (df['LapNumber'] - 1) * fuel_burn_per_lap
    
    # Calculate the time advantage gained (The "Mask")
    # As fuel burns, the car gets naturally faster. We add this time BACK 
    # to simulate what the tire would feel like if the car weight was constant.
    df['FuelCorrection'] = df['FuelBurned'] * time_loss_per_kg
    
    df['FuelCorrectedTime'] = df['LapTimeSeconds'] + df['FuelCorrection']
    
    return df
