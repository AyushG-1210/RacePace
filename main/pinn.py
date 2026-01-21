import torch
import torch.nn as nn

class TirePINN(nn.Module):
    def __init__(self, num_drivers=20, num_circuits=25):
        super().__init__()
        
        # 1. Embeddings: Turn IDs into meaningful vectors
        # (Input: 1 ID -> Output: Vector of size 4)
        self.driver_embed = nn.Embedding(num_drivers, 4)
        self.circuit_embed = nn.Embedding(num_circuits, 4)
        
        # 2. Main Network
        # Input Size: 
        #   1 (Age) + 1 (Temp) + 3 (Compounds) + 4 (Driver Vec) + 4 (Circuit Vec) = 13
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.Tanh(),  # <--- Crucial for Physics
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1) # Output: Predicted Pace
        )
        
    def forward(self, x_continuous, x_driver_id, x_circuit_id):
        """
        x_continuous: [Batch, 5] -> (Age, Temp, Soft, Med, Hard)
        x_driver_id:  [Batch, 1] -> (e.g., 1 for VER)
        x_circuit_id: [Batch, 1] -> (e.g., 5 for Zandvoort)
        """
        # 1. Lookup the embeddings
        drv_vec = self.driver_embed(x_driver_id).squeeze(1)
        ckt_vec = self.circuit_embed(x_circuit_id).squeeze(1)
        
        # 2. Combine everything
        combined_input = torch.cat([x_continuous, drv_vec, ckt_vec], dim=1)
        
        # 3. Predict
        return self.net(combined_input)