import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import your components
from get_data import run_ingestion
from analyze_style import StyleAnalyst
from package_data import DataPackager
from pinn_model import UndercutPINN # Assuming you saved the class above in pinn_model.py

# --- CONFIGURATION ---
EPOCHS = 500
LEARNING_RATE = 0.001

def train_model():
    # 1. PREPARE DATA (The Pipeline)
    print("--- 1. DATA PIPELINE ---")
    data_bundle = run_ingestion() # User inputs: 2024, Monaco, PIA, VER
    
    # Clean & Analyze
    analyst = StyleAnalyst(data_bundle)
    blueprint = analyst.run()
    clean_laps = analyst.clean_laps
    weather = data_bundle['race']['weather']
    
    # Package for PyTorch
    packager = DataPackager()
    df_pinn, feature_cols, target_col = packager.process(clean_laps, weather)
    
    # Convert to Tensors
    X_train = torch.tensor(df_pinn[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(df_pinn[target_col].values, dtype=torch.float32)
    
    # Create Loader (Batching helps training stability)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. INITIALIZE MODEL
    print("\n--- 2. INITIALIZING PINN ---")
    model = UndercutPINN(blueprint)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # We use MSE (Mean Squared Error) for the Data Loss
    criterion = nn.MSELoss()
    
    # 3. TRAINING LOOP
    print(f"\n--- 3. TRAINING ({EPOCHS} Epochs) ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            
            # A. Forward Pass
            # Note: For training, we don't run the 4-lap battle loop.
            # We just want to predict the "Instantaneous Pace" for this specific data point.
            
            # We bypass the 'forward' battle loop and access components directly 
            # to train the "Tire Brain" on single laps first.
            grip = model.tire_net(batch_X)
            
            # We assume constant mass for training batches if we normalized inputs
            # Or better: Extract mass from the batch if we added it as a feature.
            # For simplicity here, we use the average mass of the stint.
            avg_mass = blueprint['coefficients']['min_mass_kg'] + 20 # Placeholder
            
            # Predict Time using Physics Layer
            pred_time = model.physics_law(grip, avg_mass, base_pace=0.5) # 0.5 is normalized "mid-pace"
            
            # B. Loss Calculation
            # 1. Data Loss: (Prediction - Actual)^2
            data_loss = criterion(pred_time, batch_y)
            
            # 2. Physics Regularization (The "Informed" part)
            # Penalize if Grip is negative (Impossible) or > 1.5 (Unrealistic)
            physics_loss = torch.relu(-grip).sum() + torch.relu(grip - 1.5).sum()
            
            # Combine
            loss = data_loss + (0.1 * physics_loss)
            
            # C. Backpropagation (The Learning)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.6f}")
            
    # 4. SAVE & VALIDATE
    print("\n--- 4. VALIDATION ---")
    model.eval()
    
    # Denormalize a sample prediction to see if it makes sense in seconds
    with torch.no_grad():
        test_grip = model.tire_net(X_train[0:1])
        test_pred = model.physics_law(test_grip, avg_mass, base_pace=0.5)
        
        # Invert scaling
        actual_time = packager.scalers['LapTime'].inverse_transform(y_train[0:1].reshape(-1, 1))
        pred_time_sec = packager.scalers['LapTime'].inverse_transform(test_pred.reshape(-1, 1))
        
        print(f"Sample Actual:    {actual_time[0][0]:.3f} s")
        print(f"Sample Predicted: {pred_time_sec[0][0]:.3f} s")
        
    # Save the calibrated brain
    torch.save(model.state_dict(), 'hero_model.pth')
    print("Model Saved to 'hero_model.pth'")

if __name__ == "__main__":
    train_model()