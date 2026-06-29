# F1 PINN Architecture Overhaul: Change Log & Physics Diff

This update synchronizes the **Micro-Telemetry** (10,000+ points/lap) with the **Physical Constraints** of the car. The primary objective was resolving unit mismatches where the model processed normalized decimals ($0.0-1.0$) while the physics equations required SI units ($m/s$, $kg$, $N$).

---

## 🛠️ Technical Change Log

### 1. Data Ingestion & Processing (`get_data.py`, `process_data.py`)
* **Telemetry Expansion:** Race loading now requires `telemetry=True`. This provides the high-fidelity Speed, Throttle, and Brake traces necessary for calculating Physics Loss.
* **Timedelta Eradication:** All `Time` objects are converted to `RaceTimeSeconds` (float) immediately upon ingestion to prevent PyTorch tensor conversion errors.
* **Curvature Pathfinding:** Integrated a **Savitzky-Golay filter** in `StyleAnalyst`. This smooths X/Y coordinates before calculating derivatives, preventing "noisy" GPS data from triggering artificial panic-braking in the AI.
* **Regression Optimization:** Implemented `.groupby('LapNumber')` for tire degradation checks. This prevents the system from running redundant linear regressions on hundreds of thousands of duplicate telemetry points.

### 2. Feature Engineering (`data_packaging.py`)
* **Arrhenius Preservation:** Removed `MinMaxScaler` from `TyreLife`. The thermal model now receives **Raw Lap Counts** ($1, 2, 3...$), which is critical for the exponential decay math in the Arrhenius equations.
* **Fuel Correction Removal:** Deleted manual fuel-weight subtraction logic. The PINN now calculates fuel mass and the resulting time penalty natively within the `physics_law`.
* **Dynamic Weather Merge:** Utilized `pd.merge_asof` to sync every telemetry point with the exact `TrackTemp` recorded at that specific millisecond of the race.

### 3. PINN Architecture & Physics (`pinn_model.py`)
* **Unit Re-Alignment:** Added internal un-normalization for Velocity. The AI "Brain" processes $V_{norm}$, but the physics loop uses $V_{ms} = V_{norm} \times 100.0$.
* **Engine Force Integration:** Centralized `get_engine_force` using the Power/Traction limit formula: $F_{max} = \min(P/v, F_{traction})$.
* **Loss Engine Dynamics:** Updated `RacingPhysicsLoss` to pull `min_mass_kg` dynamically from the blueprint instead of using static $800kg$ estimates.

### 4. Training Stability (`train_model.py`)
* **Ghost Car Calibration:** Fixed ghost car normalization to use the $100 m/s$ ceiling, ensuring "What-If" scenarios correctly map to the model's expected velocity range.
* **Unified Physics Loss:** Replaced manual acceleration math in the training loop with the centralized `RacingPhysicsLoss` class for 1:1 parity between training and simulation.
* **Sim-Check Accuracy:** Updated the final `hero_state` to use $0.70$ (representing $70 m/s$) to prevent the "Mach 2" simulation error caused by feeding raw speeds into Tanh activations.

---

## 📉 Physics "Diff": Legacy vs. Finalized

| Feature | Legacy Logic (Previous Save) | Finalized Physics (Current) |
| :--- | :--- | :--- |
| **Velocity (Physics)** | Used $V_{norm}$ ($0.8$) in $1/2 \rho v^2$ | $V_{norm} \to V_{ms}$ ($80.0$) in $1/2 \rho v^2$ |
| **Tire Age** | Normalized ($0.0$ to $1.0$) | **Raw Lap Count** ($1, 2, 3, ...$) |
| **Braking Logic** | Sudden cut to $0$ throttle | Elastic `throttle_fade` based on $decel_{req}$ |
| **Track Curvature** | Raw/Noisy Telemetry | Smoothed via Savitzky-Golay Filter |
| **Fuel Weight** | Pre-subtracted from Lap Time | Dynamically calculated: $m_{fuel} \times 0.035s$ |
| **Engine Limits** | Hardcoded Max Force | Velocity-dependent: $Power / Velocity$ |
| **G-Force Limits** | Static 1.8G Circle | Dynamic "Mushroom" Performance Envelope |

Module 1: Architectural Integrity
- History-Free Experts: Experts now receive only "irreducible" physics features ($v$, tyre age, temp, compounds) to break the circular dependency where noisy history creates noisy grip
- Soft-Gating Transition: The Mixture of Experts (MoE) gate slope is reduced from $1000$ to $100$, allowing experts to blend over $40\text{m}$ instead of "snapping" instantly.
- Proportional Control Policy: Replaces "binary" braking with a $6$-point planning horizon, scaling brake pressure proportionally to velocity excess rather than using a fixed threshold.
- Grip Prediction EMA: Implements an Exponential Moving Average ($\alpha=0.75$) on predicted $\mu$ within the simulation to decouple control decisions from single-step network noise.

Module 2: Physics Engine & PDEG-Unit Normalization: All forces in the PDE are divided by $(Mass \times 9.81)$, scaling residuals to the $O(1)$ range so they don't drown out smoothing terms.
- History Consistency Residual: A PDE term that penalizes jumps between $\mu_t$ and $\mu_{t-1}$ using the rolling history buffer as fixed, observed anchors.
- Squared "Center-Pull" Regularizer: Replaces the linear reward with a squared penalty pulling grip toward $1.45\mu$ to prevent the model from "hiding" at the $1.2$ floor.
- Geometric Velocity Anchor: A "Ghost Guide" residual that penalizes the car for driving slower than the theoretical limit defined by $V \approx \sqrt{1.4 \cdot g / \kappa}$.

Module 3: Stable-Physics (Game Engine Logic)
- Temporal Sub-stepping (RK4 Style): The PDE evaluates physics at both the current point and an internal "mid-point" lookahead to find a smoother gradient across "stiff" transitions.
- Warm-starting (Temporal Coherence): Penalizes radical shifts in the neural network's internal Hidden Layer activations between adjacent meters to ensure "logical consistency."
- Physical Damping (Slew-Rate Penalty): Adds a derivative penalty to the rate of change of throttle and brake ($\frac{dControl}{ds}$), mathematically acting like "honey" on the pedals.

Module 4: Training & Convergence
- Hard-Clamped Experts: The final layer uses $1.2 + \text{Softplus}(x) \times 0.5$, making it physically impossible for the network to output a value outside the high-performance range.
- Learning Rate Decay: Implements a StepLR scheduler to drop the learning rate from $0.001$ to $0.0001$, preventing "optimization ringing" (the bunny-hop) as the model nears a solution.
- L-BFGS Phase 2: A second-order optimization phase with increased iterations ($5000$) to "iron out" the remaining micro-jitters using exact curvature math.
