import torch
import torch.nn as nn

class UndercutPINN(nn.Module):
    def __init__(self, physics_blueprint):
        super().__init__()
        
        # 1. Load Physics Constants from your Blueprint
        # (These are fixed facts, not things the network learns)
        self.base_mass = physics_blueprint['coefficients']['min_mass_kg']
        self.fuel_burn = physics_blueprint['coefficients']['fuel_burn_per_lap']
        self.pit_loss = 22.5 # Average pit loss (seconds)
        
        # 2. The "Tire Brain" (Neural Network)
        # It predicts ONLY the unknown variable: The Tire Friction Coefficient (mu)
        # Input: [TireAge, Compound_OneHot(3), TrackTemp] = 5 inputs
        self.tire_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(), # Tanh is best for physics (smooth gradients)
            nn.Linear(32, 16),
            nn.Softplus(), # Enforces positive output (Friction can't be negative)
            nn.Linear(16, 1)  # Output: Grip Factor (0.0 to 1.2)
        )
        
    def physics_law(self, grip_factor, mass, base_pace):
        """
        The Hardcoded Layer: Converts Grip & Mass into Lap Time.
        Formula: Time = Base + (Drag/Grip) + (Mass * Penalty)
        """
        # A simple physics approximation for lap time:
        # Less grip = Higher time (inversely proportional)
        grip_penalty = 1.0 / (grip_factor + 1e-6) 
        
        # More mass = Higher time (F=ma)
        mass_penalty = (mass - self.base_mass) * 0.035 # 0.035s per kg
        
        predicted_lap_time = base_pace + grip_penalty + mass_penalty
        return predicted_lap_time

    def forward(self, x_state, rival_pace_avg, current_gap):
        """
        Simulates the next 4 Laps to find the Undercut Window.
        
        x_state: [Batch, 5] -> Initial Condition (Age, Temp, Soft, Med, Hard)
        rival_pace_avg: Scalar (e.g., 80.5 seconds)
        current_gap: Scalar (e.g., +2.0 seconds)
        """
        
        predictions = []
        cumulative_time = 0.0
        gap_trajectory = []
        
        # We clone the state so we can mutate it (simulate future laps)
        current_state = x_state.clone()
        current_mass = self.base_mass + (self.fuel_burn * 10) # Assume 10 laps fuel for undercut push
        
        # --- THE BATTLE LOOP (Simulate 4 Laps) ---
        for lap in range(4):
            # 1. Ask the Neural Net for current Grip
            grip = self.tire_net(current_state)
            
            # 2. Apply Physics Layer to get Lap Time
            # We assume 'Base Pace' is 75s (approx pole time) - trainable bias
            hero_lap_time = self.physics_law(grip, current_mass, base_pace=75.0)
            
            # 3. Calculate Logic (The Battle)
            # Lap 0 is the OUTLAP (We add Pit Loss here)
            if lap == 0:
                hero_lap_time += self.pit_loss
            
            # Update the Gap
            # New Gap = Old Gap - (Rival Time - Hero Time)
            # (Note: Rival is on old tires, so we assume constant slow pace for simplicity)
            current_gap = current_gap - (rival_pace_avg - hero_lap_time)
            gap_trajectory.append(current_gap)
            
            # 4. Update State for NEXT lap (Time Marching)
            # Increase Tire Age by 1.0
            current_state[:, 0] += 1.0 
            # Decrease Mass (Burn Fuel)
            current_mass -= self.fuel_burn
            
        # Stack results
        # output: [Gap_Lap1, Gap_Lap2, Gap_Lap3, Gap_Lap4]
        return torch.stack(gap_trajectory, dim=1)

    def predict_probability(self, gap_trajectory):
        """
        Converts the Gap Trajectory into a % chance of success.
        """
        # If Gap < 0 at any point, undercut is possible.
        min_gap = torch.min(gap_trajectory)
        
        # Sigmoid function to turn Gap into Probability
        # If gap is -1.0s, Prob ~ 90%
        # If gap is +1.0s, Prob ~ 10%
        probability = torch.sigmoid(-min_gap * 2.0) 
        return probability.item() * 100

# --- INSTANTIATE ---
# Fake blueprint for testing
dummy_blueprint = {'coefficients': {'min_mass_kg': 798, 'fuel_burn_per_lap': 1.7}}
model = UndercutPINN(dummy_blueprint)

# Fake Input: [Age=0 (New Tires), Temp=0.5, Hard_Tire=1, Soft=0, Med=0]
hero_state = torch.tensor([[0.0, 0.5, 0.0, 0.0, 1.0]]) 
rival_pace = torch.tensor([[82.0]]) # Rival doing 1:22.0s
gap = torch.tensor([[22.0]]) # We are 22s behind (entering pits)

# Run Simulation
gaps = model(hero_state, rival_pace, gap)
prob = model.predict_probability(gaps)

print(f"Projected Gaps (Laps 1-4): {gaps.detach().numpy()}")
print(f"Undercut Success Chance: {prob:.1f}%")