import matplotlib.pyplot as plt
from get_data import get_race_data      # Your previous file
from process_data import add_fuel_correction # The new file

# 1. Get Data
df = get_race_data(2025, 'Zandvoort', 'PIA')

# 2. Process Data
df = add_fuel_correction(df)

# 1. Filter out Pit Stops / Safety Cars (e.g., any lap > 85 seconds)
clean_stint = df[df['LapTimeSeconds'] < 85].copy()

# 2. Let's look at just ONE stint (e.g., Laps 35 to 50)
# This is usually a clean period in the middle of the race
clean_stint = clean_stint[(clean_stint['LapNumber'] > 35) & (clean_stint['LapNumber'] < 50)]

# 3. Plot AGAIN
plt.figure(figsize=(10, 6))
plt.plot(clean_stint['LapNumber'], clean_stint['LapTimeSeconds'],
         label='Raw (Fuel Advantage Hides Deg)', color='gray', linestyle='--')
plt.plot(clean_stint['LapNumber'], clean_stint['FuelCorrectedTime'],
         label='Corrected (The True Physics)', color='red', linewidth=3)

plt.title("Zoomed In: The True Tire Decay (Laps 35-50)")
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
