# Project:
- Build a fully funtioning F1 race tire wear predictor based on SciML methods and PINN's for ODE calculations.

# Actual Flow:
1. Use FastF1 to get data, gather input from user (track, driver, session, etc).
2. Preprocess data, clean and structure it for model input.
3. Build the model using Physics Informed Neural Networks to solve the ODE's governing tire wear.
4. Update priors using Bayesian Inference, remove noisy data using Kalman Filtering, run Monte Carlo simulations to generate possible scenarios.
5. Visualize results and generate predictions for tire wear over the race, clear cache.

# Design Logic:
- Physics Informed Neural Networks for solving the ODE's that govern tire wear.
- Bayesian Interference for parameter estimation.
- Kalman Filtering for state estimation.
- Monte Carlo Simulations for uncertainty quantification.

# Refining:
- Use Automatic Mixed Precision (AMP) to speed up training times.
- Use vectorized pandas and numpy operations to optimize data processing.
- Use GPU acceleration for matrix calculations.
- Use PostGres to store database and redis to cache data.
- Use Docker to containerize the application for easy deployment.
- Use custom CI/CD for automated testing and deployment, or GitHub Actions for automated workflows.

# Tech stack and steps:
- Python for data processing, model building, and training.
- Pytorch for building nueral network.
- Bash for scripting.
- Docker for containerization.
- GitHub for version control.
- Jupyter Notebooks for experimentation and prototyping.
- Pandas and NumPy for data manipulation and numerical operations.
- Matplotlib and Seaborn for data visualization.
- Hosting platform undecided.

>### Notes:<br>
>- Focusing on 1 driver strategy, as it's not feasible to find Global Nash Equilibrium for all drivers in a race.<br>
>- Calculating tire wear only, with an assumed constant fuel consumption and controlled other factors.
>- Using ETL pipelines for data processing and model training. (Phase 1)
>- Modelling Arrhenius equation for decay law of tire wear.
>- Using game state vector to capture all data required for model output.