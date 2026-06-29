import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# 1. Simulate the "True" Path (Physics) and "Sensor" Readings (Noise)
np.random.seed(2)
time_steps = 50
dt = 1.0 # 1 second per step
velocity = 50 # m/s

true_positions = []
sensor_readings = []
current_pos = 0

for t in range(time_steps):
    # Physics: Move forward 50m
    current_pos += velocity
    true_positions.append(current_pos)
    
    # Noise: Add random garbage (+/- 300m variance)
    noise = np.random.normal(0, 300) 
    sensor_readings.append(current_pos + noise)

# 2. YOUR JOB: Configure the Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1) # 2 Variables (Pos, Vel), 1 Sensor (Pos)

# INITIAL STATE (Where do we think we are?)
kf.x = np.array([0., 0.])       # Start at 0, Velocity 0

# STATE TRANSITION MATRIX (F) - The Physics
# Pos_new = Pos_old + (Vel_old * dt)
# Vel_new = Vel_old
kf.F = np.array([[1., dt],
                 [0., 1.]])

# MEASUREMENT FUNCTION (H) - What do we measure?
# We only measure Position (1st variable), not Velocity.
kf.H = np.array([[1., 0.]])

# COVARIANCE MATRICES (The "Trust" Knobs)
# P = Prediction Uncertainty (How unsure are we at the start?)
kf.P *= 1000. 
# R = Measurement Noise (How trash is the sensor?) -> HIGH value = Trust sensor less
kf.R = 50000 
# Q = Process Noise (How much does the driver buffet around?)
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)

# 3. RUN THE FILTER
kalman_estimates = []

for z in sensor_readings:
    kf.predict()   # Step 1: Physics Guess
    kf.update(z)   # Step 2: Correct with Measurement
    kalman_estimates.append(kf.x[0]) # Store the Position estimate

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(true_positions, 'k--', label='True Path (God View)')
plt.plot(sensor_readings, 'rx', alpha=0.5, label='Noisy GPS Input')
plt.plot(kalman_estimates, 'b-', linewidth=3, label='Kalman Filter Output')
plt.legend()
plt.title("Week 2: Restoring the Truth")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid(True, alpha=0.3)
plt.show()