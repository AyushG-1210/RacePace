import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np

# Setup
fastf1.Cache.enable_cache('cache') 
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# Load Data
session = fastf1.get_session(2025, 'Zandvoort', 'R') 
session.load()

# Define Drivers
drivers = ['PIA', 'RUS']

# Plot Setup
fig, ax = plt.subplots(figsize=(12, 7))

# CONSTANTS (Zandvoort Specifics)
# 0.033s is a standard estimate for high-downforce tracks
FUEL_CORRECTION_FACTOR = 0.033 
FUEL_PER_LAP = 1.7 
START_FUEL = 110

for driver in drivers:
    # 1. Get Data
    laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index()
    
    # 2. THE PHYSICS ENGINE (Fuel Correction)
    # Calculate how much fuel was on board at every lap
    current_fuel_weight = START_FUEL - (laps['LapNumber'] * FUEL_PER_LAP)
    
    # Calculate the "Fuel Advantage" they had relative to a heavy car
    # As fuel drops, this value gets smaller? No.
    # We want to normalize to "Zero Fuel" or "Start Fuel". 
    # Let's normalize to "Zero Fuel" (Base Pace).
    # ActualTime = BaseTime + (Weight * 0.033)
    # BaseTime (Tyre Health Only) = ActualTime - (Weight * 0.033)
    
    # Wait, let's normalize to START WEIGHT.
    # If the car stayed at 110kg, how slow would it be?
    # It would be SLOWER than the actual time.
    fuel_handicap = (START_FUEL - current_fuel_weight) * FUEL_CORRECTION_FACTOR
    laps['DegradationCurve'] = laps['LapTime'].dt.total_seconds() + fuel_handicap

    # 3. Plot
    # Smooth the data slightly to hide traffic noise (Rolling Average)
    smooth_curve = laps['DegradationCurve'].rolling(window=3).mean()
    
    ax.plot(laps['LapNumber'], smooth_curve, 
            label=f"{driver} (Fuel Corrected)",
            linewidth=2,
            color=fastf1.plotting.get_driver_color(driver, session=session))

# Visuals
ax.set_ylabel("Fuel-Normalized Lap Time (s)")
ax.set_xlabel("Lap Number")
ax.set_title("True Tire Degradation Model (The 'DeepMind' View)")
ax.invert_yaxis() # Standard F1 format: Higher is Slower
ax.legend()
ax.grid(True, alpha=0.2)

# Add "Zone" annotations
plt.axvspan(40, 42, color='cyan', alpha=0.1, label='RUS Incident?')

plt.savefig('degradation_plot.png', dpi=150, bbox_inches='tight')
plt.close()