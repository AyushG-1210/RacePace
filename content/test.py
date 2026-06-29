import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import pandas as pd

# 1. Setup - Enable cache so you don't download data every time
# Create a folder called 'cache' in your project directory first
fastf1.Cache.enable_cache('cache')

# 2. Configure the plot style to look "Sci-Fi" / Dark Mode (DeepMind aesthetic)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

print("Fetching Data... (This takes a minute for the first run)")

# 3. Load the Session
# We look at the 2024 Abu Dhabi GP (or use '2023', 'Abu Dhabi', 'R' for a past known race)
# Since the race is "live" or just finished, we might need to use the specific session identifier
session = fastf1.get_session(2025, 'Zandvoort', 'R') 
session.load()

print("Data Loaded. Analyzing Pace...")

# 4. Define the Battle
driver_1 = 'PIA'
driver_2 = 'RUS'

# 5. Extract Lap Times
# pick_quicklaps() removes Safety Car laps / slow in-laps automatically
laps_d1 = session.laps.pick_driver(driver_1).pick_quicklaps().reset_index()
laps_d2 = session.laps.pick_driver(driver_2).pick_quicklaps().reset_index()

# 6. The "Research" Insight: Calculating the Gap
# We can't just plot time; we need to plot the *delta* to see the undercut
# This simple logic plots lap time evolution
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Driver 1 (Lando)
ax.plot(laps_d1['LapNumber'], laps_d1['LapTime'], 
        color=fastf1.plotting.get_driver_color(driver_1, session=session), 
        label=f"{driver_1}")

# Plot Driver 2 (Max)
ax.plot(laps_d2['LapNumber'], laps_d2['LapTime'], 
        color=fastf1.plotting.get_driver_color(driver_2, session=session), 
        label=f"{driver_2}")

# 7. Make it look professional
ax.set_ylabel("Lap Time")
ax.set_xlabel("Lap Number")
ax.invert_yaxis() # In racing, lower is better (higher on graph)
ax.set_title(f"Race Pace Analysis: {driver_1} vs {driver_2}")
ax.legend()
plt.grid(True, alpha=0.3)

# 8. Save the proof
plt.savefig("race_pace_v1.png")
print("Analysis saved as 'race_pace_v1.png'")
plt.show()