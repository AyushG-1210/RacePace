import torch
import torch.nn as nn
from data_packaging import process_packaging

class TirePINN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # INPUT: 5 Features (Age, Temp, Soft, Med, Hard)
        # OUTPUT: 1 Value (Predicted Pace)
        
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),        # Tanh is better for Physics/Derivatives than ReLU
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Output: The Predicted Time
        )
        
    def forward(self, x):
        return self.net(x)
def physics_loss(model, x_input, y_actual):
    
    # 1. Enable Derivative Tracking for the Input (Tire Age)
    x_input.requires_grad = True
    
    # 2. Forward Pass (Predict Time)
    y_pred = model(x_input)
    
    # --- LOSS A: DATA FIT (Standard MSE) ---
    data_loss = torch.mean((y_pred - y_actual) ** 2)
    
    # --- LOSS B: PHYSICS CONSTRAINT (Monotonicity) ---
    # We calculate the gradient: How much does Pace change as Age changes?
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=x_input,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]
    
    # Extract the gradient for 'TireAge' (Index 0)
    # This represents the "Degradation Rate"
    deg_rate = grads[:, 0] 
    
    # Constraint: Degradation Rate must be POSITIVE (Time goes UP)
    # If deg_rate < 0 (getting faster), we punish the model heavily.
    # ReLU(-deg_rate) gives 0 if positive, and the error value if negative.
    physics_violation = torch.relu(-deg_rate) 
    physics_loss = torch.mean(physics_violation ** 2)
    
    # 3. Combine
    # We weight physics heavily (e.g., 10.0) so the model respects it.
    total_loss = data_loss + (10.0 * physics_loss)
    
    return total_loss
