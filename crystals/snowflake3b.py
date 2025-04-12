import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import convolve
import time
from tqdm import tqdm
from numba import jit, prange
import argparse

class AdvancedSnowflakeSimulation:
    """
    An advanced model for simulating snowflake growth based on diffusion-limited aggregation
    with Turing patterns and physically accurate parameters.
    """
    
    def __init__(self, 
                 size=400, 
                 symmetry_order=6,
                 diffusion_coefficient=0.14,
                 anisotropy_strength=0.3,
                 attachment_probability=0.65,
                 vapor_density_factor=1.05,
                 temperature=-15.0,  # Celsius
                 supersaturation=0.1,
                 boundary_vapor_density=0.6,
                 external_field_strength=0.05,
                 noise_scale=0.02,
                 use_jit=True):
        """
        Initialize the advanced snowflake simulation.
        
        Parameters:
        -----------
        size : int
            Size of the grid (higher values mean more detailed snowflakes but slower simulation)
        symmetry_order : int
            Order of rotational symmetry (6 for hexagonal, 12 for dodecagonal, etc.)
        diffusion_coefficient : float
            Coefficient controlling vapor diffusion rate (higher = faster diffusion)
        anisotropy_strength : float
            Strength of crystallographic anisotropy (0=isotropic, 1=highly anisotropic)
        attachment_probability : float
            Probability of vapor molecules attaching to the crystal
        vapor_density_factor : float
            Controls the vapor density outside the crystal
        temperature : float
            Simulation temperature in Celsius (affects growth patterns)
        supersaturation : float
            Level of supersaturation in the vapor (higher values promote faster growth)
        boundary_vapor_density : float
            Vapor density at the boundaries
        external_field_strength : float
            Strength of external field (e.g., electric field effects)
        noise_scale : float
            Scale of random fluctuations
        use_jit : bool
            Whether to use Numba JIT compilation for faster calculations
        """
        self.size = size
        self.symmetry_order = symmetry_order
        self.diffusion_coefficient = diffusion_coefficient
        self.anisotropy_strength = anisotropy_strength
        self.attachment_probability = attachment_probability
        self.vapor_density_factor = vapor_density_factor
        self.temperature = temperature
        self.supersaturation = supersaturation
        self.boundary_vapor_density = boundary_vapor_density
        self.external_field_strength = external_field_strength
        self.noise_scale = noise_scale
        self.use_jit = use_jit
        
        # Center of the grid
        self.center = size // 2
        
        # Initialize vapor density field (high everywhere except at the crystal)
        self.vapor_density = np.ones((size, size)) * self.boundary_vapor_density
        
        # Initialize crystal grid (binary: 0=vapor, 1=ice)
        self.crystal = np.zeros((size, size), dtype=np.int32)
        
        # Initialize temperature field
        self.temperature_field = np.ones((size, size)) * self.temperature
        
        # Seed the center point
        self.crystal[self.center, self.center] = 1
        self.vapor_density[self.center, self.center] = 0
        
        # Precompute crystallographic directions for anisotropy
        self.compute_anisotropy_field()
        
        # Initialize latent heat release field
        self.latent_heat = np.zeros((size, size))
        
        # Initialize diffusion mask (3x3 kernel)
        self.diffusion_kernel = np.array([
            [0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        
        # Track simulation steps
        self.steps = 0
        
        # Prepare JIT-compiled functions if enabled
        if self.use_jit:
            self.jit_diffuse_step = jit(nopython=True)(self._diffuse_step_raw)
            self.jit_growth_step = jit(nopython=True)(self._growth_step_raw)
        
        # Create neighbor lookup array for faster neighborhood checks
        self.neighbors = [(-1, -1), (-1, 0), (-1, 1), 
                          (0, -1),           (0, 1), 
                          (1, -1),  (1, 0),  (1, 1)]
    
    def compute_anisotropy_field(self):
        """Compute the anisotropy field based on crystallographic directions."""
        # Create coordinate grid
        y, x = np.ogrid[-self.center:self.size-self.center, -self.center:self.size-self.center]
        
        # Convert to polar coordinates
        r = np.sqrt(x*x + y*y) + 1e-10  # avoid division by zero
        theta = np.arctan2(y, x)
        
        # Calculate anisotropy field based on symmetry order
        # This represents the preferential growth directions
        angle_factor = np.abs(np.cos(self.symmetry_order * theta))
        
        # Normalize
        angle_factor = (angle_factor - angle_factor.min()) / (angle_factor.max() - angle_factor.min())
        
        # Distance falloff (growth preference decreases with distance)
        distance_factor = 1.0 / (1.0 + 0.1 * np.sqrt(r))
        
        # Combine factors
        self.anisotropy_field = angle_factor * distance_factor
        
        # Scale by anisotropy strength parameter
        self.anisotropy_field = self.anisotropy_strength * self.anisotropy_field + (1 - self.anisotropy_strength)
    
    def _is_boundary(self, crystal):
        """Identify boundary sites where growth can occur."""
        # Create a mask of boundary sites (vapor adjacent to crystal)
        boundary = np.zeros_like(crystal)
        
        # Loop through the grid
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                # Skip if already crystal
                if crystal[i, j] == 1:
                    continue
                
                # Check if any neighbor is a crystal site
                for di, dj in self.neighbors:
                    if crystal[i + di, j + dj] == 1:
                        boundary[i, j] = 1
                        break
        
        return boundary
    
    def _diffuse_step_raw(self, vapor_density, crystal, diffusion_coef, boundary_value):
        """Raw implementation of the diffusion step (for JIT compilation)."""
        new_vapor = np.copy(vapor_density)
        
        # Apply diffusion using a discrete Laplacian
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                # Skip if it's a crystal site
                if crystal[i, j] == 1:
                    new_vapor[i, j] = 0
                    continue
                
                # Apply diffusion using 5-point stencil
                laplacian = (
                    vapor_density[i+1, j] + vapor_density[i-1, j] + 
                    vapor_density[i, j+1] + vapor_density[i, j-1] - 
                    4 * vapor_density[i, j]
                )
                
                new_vapor[i, j] = vapor_density[i, j] + diffusion_coef * laplacian
        
        # Apply boundary conditions
        new_vapor[0, :] = boundary_value
        new_vapor[-1, :] = boundary_value
        new_vapor[:, 0] = boundary_value
        new_vapor[:, -1] = boundary_value
        
        # Ensure crystal sites have zero vapor density
        for i in range(self.size):
            for j in range(self.size):
                if crystal[i, j] == 1:
                    new_vapor[i, j] = 0
        
        return new_vapor
    
    def _growth_step_raw(self, vapor_density, crystal, boundary_vapor, anisotropy_field, 
                          attachment_prob, noise_scale, rng_seeds):
        """Raw implementation of the growth step (for JIT compilation)."""
        new_crystal = np.copy(crystal)
        
        # Find boundary sites
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                # Skip if already crystal or not at boundary
                if crystal[i, j] == 1:
                    continue
                
                # Check if it's a boundary site
                is_boundary = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if crystal[i + di, j + dj] == 1:
                        is_boundary = True
                        break
                
                if not is_boundary:
                    continue
                
                # Calculate growth probability based on various factors
                base_prob = vapor_density[i, j] * attachment_prob
                
                # Apply anisotropy effect
                direction_factor = anisotropy_field[i, j]
                
                # Apply noise
                noise = (rng_seeds[i, j] * 2 - 1) * noise_scale
                
                # Combined probability
                growth_prob = base_prob * direction_factor * (1 + noise)
                
                # Probabilistic growth
                if rng_seeds[i, j + self.size] < growth_prob:
                    new_crystal[i, j] = 1
        
        return new_crystal
    
    def diffuse_vapor(self):
        """Diffuse water vapor around the crystal."""
        if self.use_jit:
            self.vapor_density = self.jit_diffuse_step(
                self.vapor_density, self.crystal, 
                self.diffusion_coefficient, self.boundary_vapor_density
            )
        else:
            # Create mask for non-crystal sites
            vapor_mask = 1 - self.crystal
            
            # Apply diffusion using convolution
            diffused = convolve(self.vapor_density, self.diffusion_kernel, mode='constant', cval=self.boundary_vapor_density)
            
            # Only update vapor in non-crystal sites
            self.vapor_density = self.vapor_density * (1 - vapor_mask) + diffused * vapor_mask
            
            # Enforce boundary conditions
            self.vapor_density[0, :] = self.boundary_vapor_density
            self.vapor_density[-1, :] = self.boundary_vapor_density
            self.vapor_density[:, 0] = self.boundary_vapor_density
            self.vapor_density[:, -1] = self.boundary_vapor_density
    
    def grow_crystal(self):
        """Grow the crystal based on local conditions."""
        # Find the boundary sites where growth can occur
        boundary = self._is_boundary(self.crystal)
        
        if self.use_jit:
            # Generate random seeds for JIT function
            rng_seeds = np.random.random((self.size, 2 * self.size))
            
            # Apply JIT-compiled growth step
            self.crystal = self.jit_growth_step(
                self.vapor_density, self.crystal, 
                self.boundary_vapor_density, self.anisotropy_field,
                self.attachment_probability, self.noise_scale, rng_seeds
            )
        else:
            # Calculate growth probabilities for boundary sites
            growth_prob = (
                boundary * 
                self.vapor_density * 
                self.attachment_probability * 
                self.anisotropy_field * 
                (1 + np.random.normal(0, self.noise_scale, (self.size, self.size)))
            )
            
            # Probabilistic growth
            growth = (np.random.random((self.size, self.size)) < growth_prob).astype(np.int32)
            
            # Update the crystal
            self.crystal = np.maximum(self.crystal, growth)
        
        # Update vapor density at crystal sites to zero
        self.vapor_density *= (1 - self.crystal)
    
    def update_thermal_field(self):
        """Update the thermal field based on latent heat release."""
        # Identify new crystal sites
        new_sites = self.crystal * (1 - np.roll(self.crystal, 1, axis=0))
        
        # Latent heat release at new crystal sites
        self.latent_heat = new_sites * 2260  # J/g, latent heat of solidification
        
        # Diffuse heat
        self.temperature_field += convolve(self.latent_heat, self.diffusion_kernel, mode='constant', cval=0)
        
        # Heat dissipation
        self.temperature_field = self.temperature_field * 0.99 + self.temperature * 0.01
    
    def apply_external_fields(self):
        """Apply external fields like gravity or electric fields."""
        if self.external_field_strength > 0:
            # Simulate an external field (e.g., electric field)
            y, x = np.ogrid[-self.center:self.size-self.center, -self.center:self.size-self.center]
            field_direction = np.arctan2(y, x)  # direction from center
            
            # Modify vapor density based on field
            field_effect = np.sin(field_direction) * self.external_field_strength
            self.vapor_density *= (1 + field_effect)
            
            # Normalize vapor density
            self.vapor_density = np.clip(self.vapor_density, 0, 1)
    
    def apply_symmetry(self):
        """Apply n-fold symmetry to the crystal."""
        if self.symmetry_order <= 1:
            return  # No symmetry to apply
        
        # Create a fresh crystal grid
        symmetric_crystal = np.zeros_like(self.crystal)
        
        # Get coordinates relative to center
        y, x = np.ogrid[-self.center:self.size-self.center, -self.center:self.size-self.center]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # First, map all points to the first sector
        sector_angle = 2 * np.pi / self.symmetry_order
        first_sector_theta = theta % sector_angle
        
        # Convert back to Cartesian coordinates
        x_first_sector = r * np.cos(first_sector_theta)
        y_first_sector = r * np.sin(first_sector_theta)
        
        # Round to integer indices
        ix_first = np.round(x_first_sector + self.center).astype(int)
        iy_first = np.round(y_first_sector + self.center).astype(int)
        
        # Clip to valid range
        ix_first = np.clip(ix_first, 0, self.size - 1)
        iy_first = np.clip(iy_first, 0, self.size - 1)
        
        # Get the crystal state in the first sector
        # (This is a bit inefficient but works for demo purposes)
        first_sector_crystal = np.zeros_like(self.crystal)
        for i in range(self.size):
            for j in range(self.size):
                if 0 <= iy_first[i, j] < self.size and 0 <= ix_first[i, j] < self.size:
                    first_sector_crystal[i, j] = self.crystal[iy_first[i, j], ix_first[i, j]]
        
        # Apply rotation symmetry
        for k in range(self.symmetry_order):
            angle = k * sector_angle
            rot_x = r * np.cos(theta + angle)
            rot_y = r * np.sin(theta + angle)
            
            # Convert to indices
            ix = np.round(rot_x + self.center).astype(int)
            iy = np.round(rot_y + self.center).astype(int)
            
            # Copy the first sector pattern to each rotated sector
            for i in range(self.size):
                for j in range(self.size):
                    if (0 <= iy[i, j] < self.size and 0 <= ix[i, j] < self.size and
                        first_sector_crystal[i, j] > 0):
                        symmetric_crystal[iy[i, j], ix[i, j]] = 1
        
        # Ensure the center is crystallized
        symmetric_crystal[self.center, self.center] = 1
        
        # Update the crystal
        self.crystal = symmetric_crystal
        
        # Update vapor density to match crystal
        self.vapor_density = self.vapor_density * (1 - self.crystal)
    
    def step(self):
        """Perform one step of the simulation."""
        # Diffuse vapor
        self.diffuse_vapor()
        
        # Grow crystal
        self.grow_crystal()
        
        # Apply symmetry
        self.apply_symmetry()
        
        # Update thermal field
        self.update_thermal_field()
        
        # Apply external fields
        self.apply_external_fields()
        
        # Increment step counter
        self.steps += 1
    
    def run(self, steps=100, progress=True):
        """Run the simulation for a given number of steps."""
        if progress:
            iterator = tqdm(range(steps), desc="Simulating snowflake growth")
        else:
            iterator = range(steps)
            
        for _ in iterator:
            self.step()
        
        return self.crystal
    
    def plot(self, ax=None, cmap='Blues', dpi=100, show_vapor=False):
        """Plot the current state of the simulation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        
        if show_vapor:
            # Plot both crystal and vapor density
            # Create a composite image
            composite = np.zeros((self.size, self.size, 4))  # RGBA
            
            # Set crystal (white with full opacity where crystal exists)
            composite[self.crystal > 0, :] = [1, 1, 1, 1]
            
            # Set vapor (blue with opacity proportional to density)
            vapor_mask = self.crystal < 1
            composite[vapor_mask, 0] = 0.7 - self.vapor_density[vapor_mask] * 0.7  # R
            composite[vapor_mask, 1] = 0.8 - self.vapor_density[vapor_mask] * 0.4  # G
            composite[vapor_mask, 2] = 1.0  # B
            composite[vapor_mask, 3] = self.vapor_density[vapor_mask] * 0.8  # A
            
            ax.imshow(composite, origin='lower')
        else:
            # Just plot the crystal
            # Create a custom colormap for the snowflake
            colors = [(0.8, 0.9, 1, 0), (0.8, 0.9, 1, 0.3), (1, 1, 1, 1)]
            custom_cmap = LinearSegmentedColormap.from_list("snowflake_crystal", colors)
            
            # Add a glow effect with Gaussian filter
            from scipy.ndimage import gaussian_filter
            glow = gaussian_filter(self.crystal.astype(float), sigma=1)
            
            # Normalize the glow
            glow = glow / np.max(glow) if np.max(glow) > 0 else glow
            
            # Create a composite image with the crystal and glow
            composite = np.zeros((self.size, self.size))
            composite = np.maximum(self.crystal, glow * 0.7)
            
            ax.imshow(composite, cmap=custom_cmap, origin='lower')
        
        # Set title with simulation parameters
        title = f"Snowflake Simulation (Steps: {self.steps})\n"
        title += f"T={self.temperature}°C, Sym={self.symmetry_order}, Aniso={self.anisotropy_strength:.2f}"
        ax.set_title(title)
        
        # Remove axes
        ax.axis('off')
        
        return ax

    def create_animation(self, total_steps=100, interval=50, save_path=None, dpi=100, show_vapor=False):
        """Create an animation of the snowflake growth."""
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        
        # Initial plot
        if show_vapor:
            # Initialize a composite image for crystal and vapor
            composite = np.zeros((self.size, self.size, 4))  # RGBA
            img = ax.imshow(composite, origin='lower', animated=True)
        else:
            # Just initialize with the crystal
            img = ax.imshow(np.zeros_like(self.crystal), 
                           cmap='Blues', origin='lower', animated=True)
        
        # Remove axes
        ax.axis('off')
        
        # Set title
        title = ax.set_title(f"Snowflake Growth (Step 0/{total_steps})")
        
        def update(frame):
            # Update the simulation
            self.step()
            
            if show_vapor:
                # Update composite image
                composite = np.zeros((self.size, self.size, 4))  # RGBA
                
                # Set crystal (white with full opacity where crystal exists)
                composite[self.crystal > 0, :] = [1, 1, 1, 1]
                
                # Set vapor (blue with opacity proportional to density)
                vapor_mask = self.crystal < 1
                composite[vapor_mask, 0] = 0.7 - self.vapor_density[vapor_mask] * 0.7  # R
                composite[vapor_mask, 1] = 0.8 - self.vapor_density[vapor_mask] * 0.4  # G
                composite[vapor_mask, 2] = 1.0  # B
                composite[vapor_mask, 3] = self.vapor_density[vapor_mask] * 0.8  # A
                
                img.set_array(composite)
            else:
                # Add a glow effect with Gaussian filter
                from scipy.ndimage import gaussian_filter
                glow = gaussian_filter(self.crystal.astype(float), sigma=1)
                
                # Normalize the glow
                glow = glow / np.max(glow) if np.max(glow) > 0 else glow
                
                # Create a composite image with the crystal and glow
                composite = np.zeros((self.size, self.size))
                composite = np.maximum(self.crystal, glow * 0.7)
                
                # Update the image
                img.set_array(composite)
            
            # Update title
            title.set_text(f"Snowflake Growth (Step {frame+1}/{total_steps})")
            
            return [img, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=total_steps, 
                             interval=interval, blit=True)
        
        # Save animation if path is provided
        if save_path:
            anim.save(save_path, writer='pillow', fps=20, dpi=dpi)
        
        plt.tight_layout()
        return anim

def temperature_study():
    """Study the effect of temperature on snowflake morphology."""
    # Different temperature regimes lead to different snowflake types
    temperatures = [-2, -5, -10, -15, -25]
    fig, axes = plt.subplots(1, len(temperatures), figsize=(20, 5))
    
    for i, temp in enumerate(temperatures):
        # Adjust parameters based on temperature
        if temp > -5:  # Plates
            aniso = 0.1
            diff = 0.1
        elif temp > -10:  # Stellar dendrites
            aniso = 0.3
            diff = 0.12
        elif temp > -20:  # Fernlike stellar dendrites
            aniso = 0.4
            diff = 0.14
        else:  # Plates and columns
            aniso = 0.2
            diff = 0.16
        
        # Run simulation
        sim = AdvancedSnowflakeSimulation(
            size=300,
            temperature=temp,
            anisotropy_strength=aniso,
            diffusion_coefficient=diff,
            symmetry_order=6,
            attachment_probability=0.6
        )
        
        # Run for fewer steps for higher temperatures (grows faster)
        steps = 100 if temp < -15 else 70
        sim.run(steps=steps, progress=False)
        
        # Plot the result
        sim.plot(ax=axes[i])
        axes[i].set_title(f"T = {temp}°C")
    
    plt.tight_layout()
    plt.savefig("temperature_study.png", dpi=150)
    plt.show()

def symmetry_study():
    """Study the effect of symmetry order on crystal structure."""
    symmetry_orders = [3, 4, 6, 8, 12]
    fig, axes = plt.subplots(1, len(symmetry_orders), figsize=(20, 5))
    
    for i, sym in enumerate(symmetry_orders):
        # Run simulation
        sim = AdvancedSnowflakeSimulation(
            size=300,
            symmetry_order=sym,
            anisotropy_strength=0.3,
            temperature=-15,
            diffusion_coefficient=0.14,
            attachment_probability=0.6
        )
        
        sim.run(steps=80, progress=False)
        
        # Plot the result
        sim.plot(ax=axes[i])
        axes[i].set_title(f"{sym}-fold Symmetry")
    
    plt.tight_layout()
    plt.savefig("symmetry_study.png", dpi=150)
    plt.show()

def parameter_study():
    """Study the effect of different parameters on snowflake growth."""
    # Define parameters to study
    diffusion_coeffs = [0.08, 0.12, 0.16, 0.20]
    anisotropy_strengths = [0.1, 0.3, 0.6, 0.9]
    
    # Create a grid of plots
    fig, axes = plt.subplots(len(diffusion_coeffs), len(anisotropy_strengths), 
                             figsize=(20, 20))
    
    # Run simulations for each parameter combination
    for i, diff in enumerate(diffusion_coeffs):
        for j, aniso in enumerate(anisotropy_strengths):
            print(f"Running simulation {i*len(anisotropy_strengths)+j+1} of "
                  f"{len(diffusion_coeffs)*len(anisotropy_strengths)}")
            
            # Run simulation
            sim = AdvancedSnowflakeSimulation(
                size=300,
                symmetry_order=6,
                diffusion_coefficient=diff,
                anisotropy_strength=aniso,
                attachment_probability=0.6,
                temperature=-15
            )
            
            sim.run(steps=70, progress=False)
            
            # Plot the result
            sim.plot(ax=axes[i, j])
            axes[i, j].set_title(f"Diff={diff}, Aniso={aniso}")
    
    plt.tight_layout()
    plt.savefig("parameter_study.png", dpi=150)
    plt.show()

def main():
    """Main function to run the snowflake simulation."""
    parser = argparse.ArgumentParser(description="Advanced Snowflake Simulation")
    parser.add_argument("--size", type=int, default=400, help="Size of the simulation grid")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--temp", type=float, default=-15, help="Temperature in Celsius")
    parser.add_argument("--sym", type=int, default=6, help="Symmetry order")
    parser.add_argument("--anis", type=float, default=0.3, help="Anisotropy strength")
    parser.add_argument("--diff", type=float, default=0.14, help="Diffusion coefficient")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--output", type=str, default="snowflake.png", help="Output image path")
    parser.add_argument("--study", type=str, choices=["temp", "sym", "param"], 
                        help="Run a parameter study")
    
    args = parser.parse_args()
    
    # Run parameter studies if requested
    if args.study == "temp":
        temperature_study()
        return
    elif args.study == "sym":
        symmetry_study()
        return
    elif args.study == "param":
        parameter_study()
        return
    
    # Create the simulation
    print(f"Initializing simulation (grid size: {args.size}x{args.size})")
    sim = AdvancedSnowflakeSimulation(
        size=args.size,
        symmetry_order=args.sym,
        diffusion_coefficient=args.diff,
        anisotropy_strength=args.anis,
        temperature=args.temp
    )
    
    # Run the simulation
    print(f"Running simulation for {args.steps} steps")
    start_time = time.time()
    sim.run(steps=args.steps)
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")
    
    # Create animation if requested
    if args.animate:
        output_anim = args.output.replace(".png", ".gif")
        print(f"Creating animation: {output_anim}")
        anim = sim.create_animation(total_steps=50, save_path=output_anim)
        plt.close()
    
    # Plot final state
    print(f"Creating final image: {args.output}")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    sim.plot(ax=ax)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    # Uncomment one of these to run different simulations
    main()
    # temperature_study()
    # symmetry_study()
    # parameter_study()
