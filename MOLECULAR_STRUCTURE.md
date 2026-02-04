# Molecular Structure and Crystal Growth Anisotropy

Crystal growth is not an isotropic process. The internal arrangement of atoms or molecules in a crystal lattice determines its "habit" or external shape. This document explains how molecular structure limits available energy states and dictates growth patterns.

## 1. Lattice Symmetry and Energy Landscapes

At the atomic scale, a crystal is a periodic arrangement of particles. The strength of bonds varies depending on the crystallographic direction.
- **High-density planes**: Planes with a high density of atoms (low Miller indices) usually have lower surface energy and grow more slowly, eventually becoming the dominant faces of the crystal.
- **Energy States**: Atoms attaching to the crystal surface seek the lowest energy state. Positions like "kinks" or "steps" on a surface provide more neighbors for a new atom, making attachment energetically favorable.

## 2. Growth Modes and Anisotropy

In our simulations, we model this molecular preference using an **anisotropy factor**.

### Hexagonal Symmetry (Ice / Snowflakes)
Ice (Ih) has a hexagonal molecular structure. Water molecules form hydrogen bonds in a 6-fold symmetric pattern.
- **Energy Landscapes**: It is much easier for molecules to attach at the "corners" of the hexagon where the number of possible hydrogen bonds is maximized.
- **Simulation**: We use a $\cos(6\theta)$ modulation to the growth rate to favor these six directions.

### Cubic Symmetry (Pyrite / Halite)
Minerals like Pyrite (Fool's Gold) or Halite (Salt) have a cubic lattice.
- **Growth preference**: Growth is strongest along the [100], [010], and [001] axes.
- **Result**: Perfect cubes or octahedral shapes emerge depending on the relative growth rates of different faces.

## 3. Diffusion-Limited Aggregation (DLA) vs. Reaction-Limited Growth

- **Diffusion-Limited**: Patterns are governed by how fast solute can reach the crystal (leading to dendritic/branching patterns like snowflakes).
- **Reaction-Limited**: Patterns are governed by the attachment kinetics at the interface (leading to smooth facets and geometric shapes).

By adjusting the "Anisotropy" and "Feed Rate" in the simulations, you are effectively changing the balance between these molecular energy constraints and the environmental transport conditions.
