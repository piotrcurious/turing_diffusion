# Electroplating Simulation Model Design

## 1. Governing Equations
The simulation will transition from a Gray-Scott reaction-diffusion model to a physically grounded electrochemical model.

### 1.1 Nernst-Planck Equation (Mass Transport)
The flux of ions ($J_i$) in the electrolyte is given by:
$$J_i = -D_i \nabla C_i - z_i u_{m,i} F C_i \nabla \phi$$
Where:
- $C_i$: Concentration of species $i$ (e.g., $Cu^{2+}$)
- $D_i$: Diffusion coefficient
- $z_i$: Charge number
- $u_{m,i}$: Ionic mobility (related to $D_i$ by Nernst-Einstein relation: $u_{m,i} = D_i / RT$)
- $\phi$: Electric potential
- $F$: Faraday constant

### 1.2 Poisson Equation (Electric Potential)
The potential distribution is determined by:
$$\nabla^2 \phi = -\frac{\rho}{\epsilon} = -\frac{F}{\epsilon} \sum z_i C_i$$
*Note: For simplicity in a real-time visualizer, we may assume electroneutrality in the bulk and solve Laplace's equation $\nabla^2 \phi = 0$ for the primary/secondary current distribution.*

### 1.3 Butler-Volmer Equation (Electrode Kinetics)
The reaction rate at the electrode surface (where deposition occurs) is governed by:
$$j = j_0 \left[ \exp\left( \frac{\alpha_a z F \eta}{RT} \right) - \exp\left( -\frac{\alpha_c z F \eta}{RT} \right) \right]$$
Where:
- $j$: Current density
- $j_0$: Exchange current density (proportional to concentration)
- $\eta$: Overpotential ($\phi_{electrode} - \phi_{solution} - E_{eq}$)
- $\alpha_a, \alpha_c$: Transfer coefficients

## 2. Numerical Implementation
- **Grid**: 2D Cartesian grid.
- **State**: 
    - `C`: Concentration of $Cu^{2+}$ ions.
    - `Phi`: Electric potential.
    - `Solid`: Binary or phase-field representing the deposited metal.
- **Update Loop**:
    1. Solve for `Phi` (Laplace/Poisson) using SOR or Multigrid.
    2. Calculate `j` at the interface using Butler-Volmer.
    3. Update `C` using Nernst-Planck (Diffusion + Migration).
    4. Grow `Solid` based on local current density `j`.

## 3. Physical Parameters (Copper Electroplating)
- $D \approx 7 \times 10^{-10} m^2/s$
- $j_0 \approx 1 - 10 A/m^2$
- $\alpha_c \approx 0.5$
- $z = 2$
