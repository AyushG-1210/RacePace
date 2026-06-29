import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class DataPackager:
    def __init__(self):
        # We keep the scalers in memory so we can un-scale the predictions later
        self.scalers = {
            'TyreLife': MinMaxScaler(),
            'TrackTemp': MinMaxScaler(),
            'LapNumber': MinMaxScaler(),
            'LapTime': MinMaxScaler() # Target variable
        }
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def identify_stints(self, df):
        """
        Assigns a unique ID to each stint.
        Logic: New Stint = Change in Compound OR Reset in TyreLife.
        """
        print("--> [PACKAGE] Identifying Stints...")
        # 1. Did the Tyre age reset? (e.g., 20 -> 1)
        df['Tyre_Reset'] = df['TyreLife'] < df['TyreLife'].shift(1)
        
        # 2. Did the compound change?
        df['Compound_Change'] = df['Compound'] != df['Compound'].shift(1)
        
        # 3. Cumulative Sum gives us the ID
        df['Stint_ID'] = (df['Tyre_Reset'] | df['Compound_Change']).cumsum().fillna(0).astype(int)
        return df

    def merge_weather_data(self, laps_df, weather_df):
        print("--> [PACKAGE] Merging Weather Data...")
        laps_df = laps_df.sort_values('Time')
        weather_df = weather_df.sort_values('Time')
        
        merged_df = pd.merge_asof(
            laps_df, 
            weather_df[['Time', 'TrackTemp', 'AirTemp']], 
            on='Time', 
            direction='backward'
        )
        return merged_df

    def add_fuel_correction(self, df, initial_fuel_kg=110, time_loss_per_kg=0.035):
        """
        Calculates the 'True' Tyre Pace by removing the fuel weight advantage.
        """
        print("--> [PACKAGE] Applying Physics (Fuel Correction)...")
        # Estimate total laps from data if possible, else default to 70
        total_laps = df['LapNumber'].max() if not df.empty else 70
        fuel_burn_per_lap = initial_fuel_kg / total_laps
        
        df['Fuel_Kg_Burned'] = (df['LapNumber'] - 1) * fuel_burn_per_lap
        fuel_penalty = df['Fuel_Kg_Burned'] * time_loss_per_kg
        
        # We create a new target column: Physics-Adjusted Pace
        df['FuelCorrectedTime'] = df['LapTimeSeconds'] + fuel_penalty
        return df

    def filter_warmup_laps(self, df):
        print("--> [PACKAGE] Removing Warmup Laps...")
        clean_stints = []
        for stint_id, group in df.groupby('Stint_ID'):
            if len(group) > 1:
                # Drop the first lap of the stint (Outlap/Warmup)
                valid_laps = group.iloc[1:].copy() 
                clean_stints.append(valid_laps)
        
        if not clean_stints:
            return df
        return pd.concat(clean_stints)

    def normalize_and_encode(self, df):
        """
        Prepares the numerical matrices for the Neural Network.
        """
        print("--> [PACKAGE] Normalizing & Encoding for PINN...")
        
        # 1. Normalize Continuous Variables (Inputs)
        # We reshape because sklearn expects 2D arrays (-1, 1)
        df['TyreLife_Norm'] = self.scalers['TyreLife'].fit_transform(df[['TyreLife']])
        df['TrackTemp_Norm'] = self.scalers['TrackTemp'].fit_transform(df[['TrackTemp']])
        df['LapNumber_Norm'] = self.scalers['LapNumber'].fit_transform(df[['LapNumber']])
        
        # 2. Normalize Target Variable (Output)
        # We need this to train, but we will Inverse Transform the output later
        df['LapTime_Norm'] = self.scalers['LapTime'].fit_transform(df[['FuelCorrectedTime']])
        
        # 3. One-Hot Encode Compounds (Categorical)
        # This creates columns like 'Compound_SOFT', 'Compound_MEDIUM'
        compound_matrix = self.encoder.fit_transform(df[['Compound']])
        compound_cols = self.encoder.get_feature_names_out(['Compound'])
        
        # Attach the OHE columns back to the dataframe
        encoded_df = pd.DataFrame(compound_matrix, columns=compound_cols, index=df.index)
        df = pd.concat([df, encoded_df], axis=1)
        
        return df

    def process(self, clean_laps, weather_data):
        # The Pipeline
        df = self.identify_stints(clean_laps)
        df = self.merge_weather_data(df, weather_data)
        df = self.add_fuel_correction(df)
        df = self.filter_warmup_laps(df)
        
        # The ML Transformation
        final_df = self.normalize_and_encode(df)
        
        # Define the exact columns the PINN will see
        # We dynamically grab the Compound_* columns we just created
        compound_cols = [c for c in final_df.columns if 'Compound_' in c]
        
        feature_cols = ['TyreLife_Norm', 'TrackTemp_Norm', 'LapNumber_Norm'] + compound_cols
        target_col = ['LapTime_Norm']
        
        return final_df, feature_cols, target_col

# --- SELF TEST ---
if __name__ == "__main__":
    from get_data import run_ingestion
    from process_data import StyleAnalyst
    
    # 1. Get Data
    data_bundle = run_ingestion()
    
    # 2. Clean (using the previous script logic)
    analyst = StyleAnalyst(data_bundle)
    clean_laps = analyst._clean_garbage_laps(data_bundle['race']['hero_laps'])
    weather = data_bundle['race']['weather']
    
    # 3. Package
    packager = DataPackager()
    df_pinn, features, target = packager.process(clean_laps, weather)
    
    print("\n[PINN INPUT MATRIX]")
    print(f"Features: {features}")
    print(df_pinn[features].head().to_string(index=False))
    
    print("\n[PINN TARGET]")
    print(df_pinn[target].head().to_string(index=False))
    
    # Verify Scaler Memory
    print(f"\n[Scaler Check] Max LapTime in Data: {packager.scalers['LapTime'].data_max_[0]:.3f}s")