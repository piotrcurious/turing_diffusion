"""
Advanced Snowflake Crystallization Simulator using FiPy
========================================================
This simulation demonstrates snowflake‐like pattern formation using a reaction–diffusion
model with anisotropic diffusion (mimicking Turing diffusion). The anisotropy is introduced by
modulating the diffusion coefficient of the activator species (u) with a six–fold symmetry.
The reaction kinetics follow a Schnakenberg model:
    f(u,v) = alpha - u + u^2*v
    g(u,v) = beta - u^2*v
The inhibitor (v) diffuses isotropically with a constant coefficient.

Run this script to see an evolving simulation and a live animation of the activator field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from fipy import (CellVariable, Grid2D, TransientTerm, DiffusionTerm,
                  Viewer, numerix)

# --------------------------
# 1. Simulation Parameters
# --------------------------
nx = 200                 # number of grid points in x
ny = 200                 # number of grid points in y
Lx = 1.0                 # domain length in x
Ly = 1.0                 # domain length in y
dx = Lx / nx            # grid spacing in x
dy = Ly / ny            # grid spacing in y

# Physical and chemical parameters
Du_base = 0.1           # base diffusion coefficient for u (activator)
Dv = 0.5                # diffusion coefficient for v (inhibitor)
epsilon = 0.3           # anisotropy strength [0,1], modulates hexagonal symmetry

# Reaction (Schnakenberg) parameters:
alpha = 0.1
beta = 0.9

dt = 0.001              # time step
total_steps = 5000      # total simulation steps
plot_interval = 50      # update visualization every this many time steps

# --------------------------
# 2. Create a 2D Mesh
# --------------------------
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# Obtain cell center coordinates for anisotropy definition
x, y = mesh.cellCenters[0], mesh.cellCenters[1]

# Compute an angle relative to the center of the domain
# (shifting the origin to center (Lx/2,Ly/2) produces radial symmetry)
theta = np.arctan2(y - Ly/2, x - Lx/2)

# Define an anisotropic diffusion coefficient for u that has sixfold symmetry.
# The modulation term cos(6*theta) imposes hexagonal preferential directions.
Du_field = Du_base * (1.0 + epsilon * np.cos(6 * theta))

# --------------------------
# 3. Define Variables and Initial Conditions
# --------------------------
# Initialize u (activator) and v (inhibitor) with a nearly uniform state plus a small random perturbation.
u = CellVariable(name="Activator", mesh=mesh, value=alpha + 0.02*(np.random.random(mesh.numberOfCells)-0.5))
v = CellVariable(name="Inhibitor", mesh=mesh, value=beta/alpha + 0.02*(np.random.random(mesh.numberOfCells)-0.5))

# --------------------------
# 4. Define the PDEs (Reaction-Diffusion Equations)
# --------------------------
# The equations are defined separately for u and v.
# For u, we incorporate anisotropic diffusion via a spatially varying coefficient (Du_field).
#   ∂u/∂t = ∇·[Du_field ∇u] + (alpha - u + u^2 * v)
# For v, we use standard isotropic diffusion.
#   ∂v/∂t = Dv ∇^2 v + (beta - u^2 * v)
eq_u = TransientTerm(var=u) == DiffusionTerm(coeff=Du_field, var=u) + (alpha - u + u**2 * v)
eq_v = TransientTerm(var=v) == DiffusionTerm(coeff=Dv, var=v) + (beta - u**2 * v)

# --------------------------
# 5. Set Up Visualization
# --------------------------
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(u.value.reshape((ny, nx)), interpolation='bilinear', cmap='viridis',
               extent=[0, Lx, 0, Ly], origin='lower')
ax.set_title("Snowflake-like Crystal Growth (Activator Field)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label='u concentration')

def update(frame):
    """Advance the simulation and update the plot."""
    for _ in range(plot_interval):
        eq_u.solve(var=u, dt=dt)
        eq_v.solve(var=v, dt=dt)
    im.set_data(u.value.reshape((ny, nx)))
    ax.set_title(f"Snowflake-like Crystal Growth (step {frame * plot_interval})")
    return [im]

anim = animation.FuncAnimation(fig, update, frames=int(total_steps/plot_interval), interval=50, blit=True)

plt.tight_layout()
plt.show()
