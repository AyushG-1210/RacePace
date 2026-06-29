import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# --- CONSTANTS (2024 Specs) ---
MIN_WEIGHT_KG = 798.0  # Car + Driver (Dry)
MAX_FUEL_KG = 110.0    # Maximum allowed fuel load

class StyleAnalyst:
    def __init__(self, data_bundle):
        """
        Input: The dictionary output from 'get_data.py'
        """
        self.q3_telemetry = data_bundle['style']['telemetry']
        self.q3_stats = data_bundle['style']['stats']
        self.race_laps = data_bundle['race']['hero_laps']
        
        # New Metadata needed for Fuel calc
        self.total_laps = data_bundle['race']['hero_laps']['LapNumber'].max()
        
        self.clean_laps = None
        
        # This is the output "Blueprint"
        self.blueprint = {
            'inputs': {},       # Interpolation functions
            'coefficients': {}  # Scalar values (Deg, Fuel, Mass)
        }

    def _clean_garbage_laps(self, df):
        """
        Filters out non-racing laps.
        """
        print(f"--> [PROCESS] Cleaning Garbage Laps (Initial: {len(df)})")
        
        # 1. Drop Lap 1 and Pit Laps
        df = df[df['LapNumber'] > 1].copy()
        df = df[pd.isna(df['PitInTime']) & pd.isna(df['PitOutTime'])].copy()
        
        # 2. 107% Rule
        median_pace = df['LapTimeSeconds'].median()
        threshold = median_pace * 1.07
        
        clean_df = df[df['LapTimeSeconds'] < threshold].copy()
        print(f"    Cleaned Count: {len(clean_df)} (Threshold: {threshold:.2f}s)")
        
        self.clean_laps = clean_df
        return clean_df

    def calculate_fuel_parameters(self):
        """
        Estimates the fuel burn rate based on race distance.
        """
        print("--> [PROCESS] Calculating Fuel & Mass Parameters...")
        
        # Assumption: Teams under-fuel slightly, but let's assume 
        # roughly 100kg-105kg for a full race distance to be safe.
        # If the race is short (e.g. Sprint), this logic adjusts automatically.
        estimated_start_fuel = 105.0 
        
        if self.total_laps > 0:
            burn_rate = estimated_start_fuel / self.total_laps
        else:
            burn_rate = 1.7 # Fallback to standard 1.7kg/lap
            
        print(f"    Est. Fuel Burn: {burn_rate:.2f} kg/lap")
        print(f"    Base Car Mass:  {MIN_WEIGHT_KG} kg")
        
        self.blueprint['coefficients']['fuel_burn_per_lap'] = burn_rate
        self.blueprint['coefficients']['min_mass_kg'] = MIN_WEIGHT_KG

    def calculate_degradation(self):
        """
        Calculates tire degradation (Linear Regression).
        """
        print("--> [PROCESS] Calculating Tire Degradation Model...")
        
        if self.clean_laps is None or len(self.clean_laps) < 5:
            print("!! [WARNING] Not enough clean laps. Using defaults.")
            self.blueprint['coefficients']['deg_per_lap'] = 0.05
            return

        # X = TyreAge, Y = LapTime
        X = self.clean_laps['TyreLife'].values.reshape(-1, 1)
        y = self.clean_laps['LapTimeSeconds'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        deg_per_lap = model.coef_[0]
        base_pace = model.intercept_
        
        print(f"    Degradation Found: {deg_per_lap:.4f} seconds/lap")
        
        self.blueprint['coefficients']['deg_per_lap'] = deg_per_lap

    def generate_control_maps(self):
        """
        Turns Q3 telemetry into continuous functions.
        """
        print("--> [PROCESS] Generating Control Input Maps (from Q3)...")
        
        dist = self.q3_telemetry['Distance'].values
        max_dist = dist.max()
        
        # Throttle: Linear interpolation
        throttle_fn = interp1d(dist, self.q3_telemetry['Throttle'] / 100.0, 
                               kind='linear', fill_value="extrapolate")
        
        # Brake: Nearest (Step function) to preserve braking points
        brake_fn = interp1d(dist, self.q3_telemetry['Brake'] > 0, 
                            kind='nearest', fill_value="extrapolate")
        
        # Gear: Nearest
        gear_fn = interp1d(dist, self.q3_telemetry['nGear'], 
                           kind='nearest', fill_value="extrapolate")

        self.blueprint['inputs']['throttle_map'] = throttle_fn
        self.blueprint['inputs']['brake_map'] = brake_fn
        self.blueprint['inputs']['gear_map'] = gear_fn
        self.blueprint['track_length'] = max_dist

    def get_car_mass(self, lap_number):
        """
        Helper function to query mass at any point in the race.
        Mass = Dry_Weight + Fuel_Remaining
        """
        burn = self.blueprint['coefficients']['fuel_burn_per_lap']
        dry = self.blueprint['coefficients']['min_mass_kg']
        
        laps_remaining = max(0, self.total_laps - lap_number)
        fuel_mass = laps_remaining * burn
        
        return dry + fuel_mass

    def run(self):
        self._clean_garbage_laps(self.race_laps)
        self.calculate_fuel_parameters() # <--- New Step
        self.calculate_degradation()
        self.generate_control_maps()
        
        # Attach the helper method reference (optional, or just use coeffs later)
        self.blueprint['methods'] = {'get_mass': self.get_car_mass}
        
        return self.blueprint

# --- SELF TEST ---
if __name__ == "__main__":
    from get_data import run_ingestion
    
    # 1. Ingest
    data = run_ingestion()
    
    # 2. Analyze
    analyst = StyleAnalyst(data)
    blueprint = analyst.run()
    
    # 3. Test Mass Calculation
    print("\n[MASS CHECK]")
    for lap in [1, 25, 50]:
        mass = analyst.get_car_mass(lap)
        print(f"Lap {lap}: {mass:.1f} kg")