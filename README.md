# RacePace: Hybrid Mixture-of-Experts Physics-Informed Neural Networks for Autonomous Vehicle Dynamics and Grip Discovery

> Department of Computer Science & Engineering  
> RV Institute of Technology and Management

## Project Status: Ongoing (In Optimization Phase)

This repository houses the foundational architecture, physical partial differential equations (PDEs), and simulation loops for **RacePace**. The system balances vehicle control policies against multi-variable aerodynamic and tire friction boundary layers using a scientific machine learning framework.

---

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![DeepXDE](https://img.shields.io/badge/DeepXDE_SciML-1.12.0-00B4D8?style=flat-square&logo=scipy&logoColor=white)
![FastF1](https://img.shields.io/badge/Data-FastF1_API-000000?style=flat-square&logo=formula1&logoColor=red)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

**RacePace** is a Physics-Informed Neural Network (PINN) architecture designed to solve transient vehicle dynamics and dynamically discover localized tire-to-track friction coefficients ($\mu$) directly from high-fidelity Formula 1 telemetry streams. 

Standard driving simulations rely on deterministic, heavily calibrated empirical tire models (such as Pacejka's Magic Formula) that fail when encountering localized track evolutions, temperature changes, or irregular tire degradation. **RacePace** approaches this problem as an inverse optimization challenge: by constraining a neural network with the fundamental equations of motion ($F=ma$), the network infers the latent, unobservable track grip capacity directly from the vehicle's observed kinematic reactions.

To prevent conflicting gradient signals during distinct track phases, the framework deploys a customized **Mixture of Experts (MoE)** architecture. This structure isolates straight-line acceleration and braking profiles from high-lateral cornering forces, allowing independent sub-networks to specialize in local aerodynamic and frictional regimes.

---

## Data Ingestion & Telemetry Pipeline

The model is trained on high-frequency, real-world Grand Prix telemetry ingested via the **FastF1 API**. 
* **State Features:** Vehicle speed, throttle application, brake pressure, and RPM sampled at 10+ Hz.
* **Track Geometry:** Local curvature ($\kappa$) dynamically derived from the second spatial derivative of GPS coordinate traces ($dx/ds, dy/ds$).
* **Physics Context:** Tire compound constraints (Soft, Medium, Hard) and degradation indices (lap stint age) are one-hot encoded and normalized to map physical states to the network.

---

## Core Mathematical Framework

RacePace maps the vehicle's state space by enforcing a continuous force-balance budget across the tire contact patches, fully coupled with aerodynamic downforce and drag variations.

### 1. The Friction Circle Boundary Constraint
The core boundary condition dictating the maximum performance envelope of the vehicle assumes that lateral and longitudinal force demands cannot exceed the available traction supplied by the normal load:

$$F_{\text{demand}} = \sqrt{F_{\text{lat}}^2 + F_{\text{long}}^2} \le \mu \cdot N$$

Where $\mu$ is the latent localized grip predicted by the neural network, and $N$ is the dynamic normal load.

### 2. Aerodynamic Coupling & Normal Load Formulation
The normal load $N$ is not static; it is heavily coupled to velocity via square-law aerodynamic downforce generation:

$$N = m \cdot g + \left( \frac{1}{2} \rho \cdot C_L \cdot A \cdot v^2 \right)$$

Where $m$ is the vehicle mass, $g$ is gravitational acceleration, $\rho$ is air density, $C_L$ is the lift/downforce coefficient, $A$ is the frontal cross-sectional area, and $v$ is the instantaneous velocity. The PDE tracking loop minimizes the residual between this maximum physical supply and the instant force demand derived from telemetry acceleration profiles.

---

## Current Architecture

The operational state of the framework is organized into isolated physics-relevant branches mapped by exact tensor dimensions to prevent data-leakage and feedback loops:

                             ┌──────────────────────────────┐
                             │    Current Physics State     │
                             │   [v, Age, Temp, Compound]   │
                             └──────────────┬───────────────┘
                                            │
                             ┌──────────────┴───────────────┐
                             │     Curvature Input (κ)      │
                             └──────────────┬───────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
            ┌──────────────────┐                            ┌──────────────────┐
            │  Straight Net    │                            │    Corner Net    │
            │ (6 Input Hidden) │                            │ (7 Input Hidden) │
            └────────┬─────────┘                            └────────┬─────────┘
                     │                                               │
                     └───────────────► MoE Gating Network ◄──────────┘
                                  Gate: σ((|κ| - 0.002) * 100)
                                            │
                                            ▼   
                                ┌───────────────────────┐
                                │ Output: 1.2 + SP * 0.3│
                                └───────────────────────┘

* **StraightExpert Net:** Processes a history-free `[Batch, 6]` tensor containing `[speed_norm, tyre_age, temp_norm, soft, med, hard]` to predict baseline straight-line performance boundaries.
* **CornerExpert Net:** Processes a `[Batch, 7]` tensor (incorporating track curvature $\kappa$) to isolate combined lateral-longitudinal tire slip friction profiles.
* **Soft-Gating Operator:** Blends the two experts using a relaxed sigmoid gate: $W = \sigma((|\kappa| - 0.002) \times 100)$. This structure forces a smooth 40-meter physical transition zone during corner entry and exit, eliminating discontinuous model switching.
* **Mathematical Range Clamping:** Output layers use a $1.2 + \text{Softplus}(x) \times 0.3$ physical constraint. In real-world Formula 1 vehicle dynamics, the longitudinal/lateral coefficient of friction ($\mu$) for slick tires strictly operates between ~1.2 (cold/worn) and ~1.5 (optimal temperature/soft). This clamping prevents the network from hallucinating hyper-physical grip ($>2.0$) to artificially minimize velocity residuals.

---

## Current Pathologies & Optimization Bottlenecks

The project is currently confronting two well-documented **Scientific Machine Learning (SciML) Gradient Pathologies** that prevent clean convergence:

### 1. The "Lazy AI" Local Minimum Trap

The model's friction prediction regularly collapses directly onto the minimum hardcoded boundary floor ($\mu \approx 1.20000x$). Because satisfying the $F=ma$ physics constraints is mathematically trivial at very low speeds, the optimizer defaults to a path of least resistance: it lowers its speed expectation to match the lowest possible grip supply, refusing to self-discover higher velocity thresholds.

### 2. The Telemetry "Bunny-Hop" Jitter

Despite high spatial smoothing penalties, the generated throttle and brake control traces exhibit extreme, high-frequency oscillations ("ringing"). This behavior is caused by a massive gradient imbalance: the core physics residuals (`res_stall_flux` and `res_v_anchor`) explode to magnitudes of $10^7$, completely drowning out the spatial smoothness residuals ($10^{-2}$). The Adam optimizer overcorrects across these massive loss valleys, forcing the car to radically cycle between full throttle and full braking every few simulated meters.

---

## Planned Improvements: The "Stable-Physics" Roadmap

To resolve these pathologies and break the optimization flatline, the next phase of development introduces three structural game-engine and multi-objective paradigms:

### 1. Temporal Sub-stepping (RK4-Style Regularization)

Instead of calculating physics derivatives solely at discrete telemetry intervals, the PDE will compute intermediate virtual states using a Runge-Kutta 4th-order approximation scheme. Evaluating the gradient at $s + \frac{\Delta s}{2}$ will provide a highly continuous, smooth loss landscape that prevents the optimizer from falling into sharp, single-point local minima.

### 2. Slew-Rate Physical Damping

To permanently eliminate the "bunny-hop" control oscillations, the PDE will incorporate a hard derivative damping tax on control changes over distance:


$$\mathcal{L}_{\text{damping}} = \left(\frac{d\text{Throttle}}{ds}\right)^2 + \left(\frac{d\text{Brake}}{ds}\right)^2$$


This functions as virtual inertia or "honey on the pedals," making high-frequency control chatter mathematically expensive for the network.

### 3. Pareto-Front Multi-Objective Equilibrium

The current strict penalty method will be refactored into a Lagrangian multi-objective loss landscape to force the model off the grip floor. The total loss will balance competing vector forces:


$$\mathcal{L}_{\text{total}} = \lambda_1 \int \frac{1}{v} ds + \lambda_2 \mathcal{L}_{\text{physics}} + \lambda_3 \mathcal{L}_{\text{damping}}$$


By treating this as a Pareto optimization problem, the network must find the exact equilibrium where the "attractive" pull of minimizing lap time ($\lambda_1$) is perfectly bounded by the "repulsive" forces of physical instability ($\lambda_2$) and driver slew-rate limits ($\lambda_3$).

---

## Target Performance & Ablation Metrics

| Optimization Methodology | Grip Discovery Accuracy ($\mu$) | Control Trace Jitter Rate | Shortcut Vulnerability |
| --- | --- | --- | --- |
| **Standard Baseline (Current)** | Flatlined at 1.200 | High ($10^7$ Oscillation) | 92.1% (Collapses to Low Speed) |
| **With RK4 + Damping (Planned)** | Target $\pm 0.02$ Error | Smooth ($<1.5\%$ variance) | < 5.0% (Stable Trajectory) |
| **Full Pareto Equilibrium (Planned)** | Optimized Real Peak | Completely Eliminated | < 0.5% (Robust Convergence) |
---

## Citation

If you utilize the AmorFlux architectural framework or system design pipelines in your research, please use the following citation format:

```bibtex
@misc{Gouda2026:RacePace,
  author       = {Ayush Gouda},
  title        = {{RacePace: Hybrid Mixture-of-Experts Physics-Informed Neural Networks for Autonomous Vehicle Dynamics and Grip Discovery}},
  howpublished = {\url{https://github.com/AyushG-1210/RacePace}},
  year         = {2026},
  month        = {August}
}
```