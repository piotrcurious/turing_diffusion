import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage as ndimage
from tqdm import tqdm
import argparse
import os
from datetime import datetime
from numba import jit


class AdvancedSnowflakeSimulation:
    def __init__(self, 
                 size=400,
                 symmetry_order=6, 
                 diffusion_coefficient=0.14,
                 attachment_rate=0.08,
                 vapor_anisotropy=0.025,
                 temperature=-15,  # Celsius
                 humidity=0.95,    # Relative humidity (0-1)
                 initial_seed_size=3,
                 boundary_vapor_density=1.0,
                 tip_bias=1.2,     # Bias growth towards pointy tips
                 random_seed=None,
                 noise_amplitude=0.02,
                 noise_frequency=0.1,
                 melting_threshold=0.98,
                 riming_probability=0.01,
                 resolution_factor=1.0):
        """
        Initialize an advanced snowflake growth simulation based on physical principles.
        
        Parameters:
        -----------
        size : int
            Size of the simulation grid (pixels)
        symmetry_order : int
            Order of rotational symmetry (6 for normal snowflakes, 12 for double plates)
        diffusion_coefficient : float
            Diffusion coefficient for water vapor
        attachment_rate : float
            Rate at which vapor attaches to the crystal
        vapor_anisotropy : float
            Degree of anisotropy in attachment (creates sharper arms)
        temperature : float
            Ambient temperature in Celsius (affects growth morphology)
        humidity : float
            Relative humidity (0-1)
        initial_seed_size : int
            Size of the initial seed crystal
        boundary_vapor_density : float
            Vapor density at the boundary of the simulation
        tip_bias : float
            Growth bias factor for tips/corners
        random_seed : int
            Seed for random number generator
        noise_amplitude : float
            Amplitude of noise in the system
        noise_frequency : float
            Spatial frequency of noise
        melting_threshold : float
            Threshold for melting/smoothing of sharp features
        riming_probability : float
            Probability of riming (attachment of supercooled droplets)
        resolution_factor : float
            Factor to increase resolution of calculations vs display
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Store parameters
        self.size = size
        self.symmetry_order = symmetry_order
        self.diffusion_coefficient = diffusion_coefficient
        self.attachment_rate = attachment_rate
        self.vapor_anisotropy = vapor_anisotropy
        self.temperature = temperature
        self.humidity = humidity
        self.initial_seed_size = initial_seed_size
        self.boundary_vapor_density = boundary_vapor_density
        self.tip_bias = tip_bias
        self.noise_amplitude = noise_amplitude
        self.noise_frequency = noise_frequency
        self.melting_threshold = melting_threshold
        self.riming_probability = riming_probability
        self.resolution_factor = resolution_factor
        
        # Compute internal parameters based on physical relationships
        self.compute_physical_parameters()
        
        # Initialize grid state
        self.init_simulation_grid()
        
        # Create lookup tables for efficiency
        self.create_lookup_tables()
        
        # Properties to track simulation state
        self.step_count = 0
        self.max_radius = initial_seed_size
        
    def compute_physical_parameters(self):
        """Compute physical parameters based on temperature, humidity, etc."""
        # Effect of temperature on growth morphology (simplified Nakaya diagram behavior)
        # -15°C tends to produce dendrites, 0°C produces plates, etc.
        
        # Temperature-dependent parameters
        if self.temperature > -3:  # Near 0°C: Thin plates
            self.basal_factor = 0.1  # Slow growth in c-axis direction
            self.prism_factor = 1.0  # Fast growth in a-axis direction
            self.branching_factor = 0.3
        elif -8 < self.temperature <= -3:  # -3 to -8°C: Needles/columns
            self.basal_factor = 1.0  # Fast growth in c-axis
            self.prism_factor = 0.1  # Slow growth in a-axis
            self.branching_factor = 0.1
        elif -18 < self.temperature <= -8:  # -8 to -18°C: Dendrites/stellar
            self.basal_factor = 0.2
            self.prism_factor = 1.0
            self.branching_factor = 1.0  # Maximum branching
        else:  # Below -18°C: Plates again
            self.basal_factor = 0.3
            self.prism_factor = 0.8
            self.branching_factor = 0.5
            
        # Scale diffusion and attachment based on temperature
        self.effective_diffusion = self.diffusion_coefficient * (1 + (self.temperature + 15) * 0.01)
        
        # Supersaturation effect (higher humidity = faster growth)
        self.supersaturation = self.humidity - 1.0  # Negative for subsaturated, positive for supersaturated
        self.growth_modifier = 1.0 + max(0, self.supersaturation * 5)  # Faster growth in supersaturated conditions
    
    def init_simulation_grid(self):
        """Initialize the simulation grids."""
        # Expand internal grid if using higher resolution
        internal_size = int(self.size * self.resolution_factor)
        
        # Center point of the grid
        self.center = internal_size // 2
        
        # Create grids:
        # - crystal: 1 where crystal exists, 0 elsewhere
        # - vapor: vapor concentration field
        # - temperature: temperature field
        self.crystal = np.zeros((internal_size, internal_size))
        self.vapor = np.ones((internal_size, internal_size)) * self.boundary_vapor_density
        self.temperature = np.ones((internal_size, internal_size)) * self.temperature
        
        # Add a seed crystal at the center
        seed_size = self.initial_seed_size
        self.crystal[self.center-seed_size:self.center+seed_size+1, 
                     self.center-seed_size:self.center+seed_size+1] = 1
                     
        # Create a perlin-like noise field for environmental variations
        self.create_noise_field(internal_size)
        
        # Enforce initial boundary conditions
        self.apply_boundary_conditions()

    def create_noise_field(self, size):
        """Create a multi-octave noise field to simulate environmental variations."""
        self.noise_field = np.zeros((size, size))
        
        # Generate multi-octave noise for more natural variations
        octaves = 4
        persistence = 0.5
        amplitude = 1.0
        
        for i in range(octaves):
            freq = self.noise_frequency * (2 ** i)
            amp = self.noise_amplitude * (persistence ** i)
            
            # Generate base noise grid at appropriate frequency
            noise_grid_size = max(2, int(size * freq))
            base_noise = np.random.randn(noise_grid_size, noise_grid_size)
            
            # Smooth the noise
            base_noise = ndimage.gaussian_filter(base_noise, sigma=1.0)
            
            # Resize to full grid
            if noise_grid_size < size:
                zoom_factor = size / noise_grid_size
                noise_layer = ndimage.zoom(base_noise, zoom_factor, order=1)
                # Ensure dimensions match
                noise_layer = noise_layer[:size, :size]
            else:
                noise_layer = base_noise
                
            self.noise_field += noise_layer * amp
            
        # Normalize noise to [-1, 1]
        self.noise_field = self.noise_field / np.max(np.abs(self.noise_field))
    
    def create_lookup_tables(self):
        """Create lookup tables for efficient computation of symmetric operations."""
        internal_size = int(self.size * self.resolution_factor)
        
        # Create distance and angle maps for efficient calculations
        y, x = np.ogrid[-self.center:internal_size-self.center, -self.center:internal_size-self.center]
        self.distance_map = np.sqrt(x*x + y*y)
        self.angle_map = np.arctan2(y, x)
        
        # Create hexagonal/symmetry masks
        self.sector_map = np.floor(self.symmetry_order * (self.angle_map + np.pi) / (2 * np.pi)) % self.symmetry_order
        
        # Create lookup for nearest neighbors - used for crystal growth dynamics
        self.neighbor_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # Create anisotropy factor map based on crystal orientation preference
        # For hexagonal ice crystals, growth is preferred along specific crystallographic directions
        self.anisotropy_map = np.zeros((internal_size, internal_size))
        
        # Different growth rates in different directions (hexagonal for typical snowflakes)
        for i in range(self.symmetry_order):
            angle = i * 2 * np.pi / self.symmetry_order
            # Regions aligned with crystal axes grow faster
            alignment = np.cos(self.symmetry_order * (self.angle_map - angle))
            self.anisotropy_map += (alignment > 0.7) * self.vapor_anisotropy
        
        # Normalize and add base value
        self.anisotropy_map = self.anisotropy_map / self.anisotropy_map.max() if self.anisotropy_map.max() > 0 else 0
        self.anisotropy_map = 1.0 + self.anisotropy_map * self.vapor_anisotropy * 10
        
        # Create a growth bias map to encourage branching
        self.tip_bias_map = np.ones((internal_size, internal_size))

    def apply_boundary_conditions(self):
        """Apply boundary conditions to the simulation."""
        internal_size = int(self.size * self.resolution_factor)
        
        # Set vapor density at boundaries
        # Use a soft boundary that depends on distance from center
        mask = self.distance_map > (0.47 * internal_size)
        self.vapor[mask] = self.boundary_vapor_density
        
        # Fix temperature at boundaries
        self.temperature[mask] = self.temperature
        
        # Also enforce high vapor density far from crystal to prevent depletion
        far_from_crystal = self.distance_map > max(internal_size * 0.3, self.max_radius * 2)
        self.vapor[far_from_crystal] = self.boundary_vapor_density
        
    def update_tip_bias_map(self):
        """
        Update the tip bias map based on current crystal geometry.
        This encourages dendritic growth by enhancing vapor attachment at tips.
        """
        # Find the crystal boundary
        crystal_boundary = np.zeros_like(self.crystal)
        crystal_dilated = ndimage.binary_dilation(self.crystal)
        crystal_boundary = np.logical_and(crystal_dilated, np.logical_not(self.crystal))
        
        # Reset tip bias map
        self.tip_bias_map = np.ones_like(self.crystal, dtype=float)
        
        # Calculate local curvature at each boundary point to identify tips
        if np.any(crystal_boundary):
            # Count neighbors for each boundary point
            boundary_pts = np.where(crystal_boundary)
            
            for i, j in zip(*boundary_pts):
                # Count crystal neighbors
                neighbor_count = 0
                for di, dj in self.neighbor_offsets:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < self.crystal.shape[0] and 
                        0 <= nj < self.crystal.shape[1] and
                        self.crystal[ni, nj] > 0):
                        neighbor_count += 1
                
                # Points with fewer crystal neighbors are more likely tips or corners
                if neighbor_count <= 2:  # Definitely a tip or protrusion
                    self.tip_bias_map[i, j] = self.tip_bias
                elif neighbor_count == 3:  # Possibly a tip
                    self.tip_bias_map[i, j] = (self.tip_bias + 1) / 2
    
    @jit(nopython=False)  # Use Numba for acceleration if available
    def diffuse_vapor(self, steps=10):
        """
        Solve the diffusion equation for vapor concentration around the crystal.
        Uses multiple substeps for stability.
        """
        # Get the mask of non-crystal regions (where diffusion happens)
        vapor_mask = (self.crystal < 0.5)
        
        # Diffusion coefficient (can vary with temperature)
        D = self.effective_diffusion
        
        # Multiple substeps for stability
        for _ in range(steps):
            # Compute Laplacian of vapor concentration (∇²c)
            laplacian = ndimage.laplace(self.vapor)
            
            # Update vapor concentration: dc/dt = D * ∇²c
            self.vapor[vapor_mask] += D * laplacian[vapor_mask]
            
            # Ensure vapor stays within physical bounds
            self.vapor = np.clip(self.vapor, 0, self.boundary_vapor_density * 1.2)
            
            # Vapor concentration is zero inside the crystal
            self.vapor[self.crystal > 0.5] = 0
            
        # Re-apply boundary conditions
        self.apply_boundary_conditions()
    
    def grow_crystal(self):
        """
        Grow the crystal based on vapor concentration, anisotropy, and other factors.
        This implements the 'reaction' part of the reaction-diffusion system.
        """
        # Find the boundary of the crystal where growth occurs
        crystal_dilated = ndimage.binary_dilation(self.crystal)
        growth_sites = np.logical_and(crystal_dilated, np.logical_not(self.crystal))
        
        # Calculate growth probability at each boundary site
        growth_probability = np.zeros_like(self.crystal)
        
        # Growth depends on:
        # 1. Local vapor concentration
        # 2. Anisotropy factor (crystal orientation preference)
        # 3. Tip bias (enhanced growth at corners/tips)
        # 4. Noise (random fluctuations)
        # 5. Temperature effects
        
        # Sample sites where growth can occur
        growth_site_indices = np.where(growth_sites)
        
        # Calculate growth probability for each site
        for i, j in zip(*growth_site_indices):
            # Base probability from vapor concentration
            prob = self.vapor[i, j] * self.attachment_rate
            
            # Apply anisotropy factor based on crystal orientation
            prob *= self.anisotropy_map[i, j]
            
            # Apply tip bias to encourage dendritic growth
            prob *= self.tip_bias_map[i, j]
            
            # Apply noise factor for random variations
            local_noise = 1.0 + 0.5 * self.noise_field[i, j]
            prob *= local_noise
            
            # Apply temperature-dependent branching factor
            r = np.sqrt((i - self.center)**2 + (j - self.center)**2)
            radial_factor = min(1.0, r / (self.size/8))  # More branching away from center
            prob *= 1.0 + (self.branching_factor * radial_factor)
            
            # Store the probability
            growth_probability[i, j] = prob
        
        # Apply growth probability to create new crystal sites
        growth_mask = (np.random.random(self.crystal.shape) < growth_probability) & growth_sites
        self.crystal[growth_mask] = 1.0
        
        # Apply riming effect (supercooled water droplets that freeze on contact)
        # More common in certain temperature ranges and humidity conditions
        if self.temperature > -8 and self.humidity > 0.8:
            rime_mask = (np.random.random(self.crystal.shape) < self.riming_probability) & growth_sites
            self.crystal[rime_mask] = 1.0
        
        # Track the maximum radius of the crystal for boundary condition management
        crystal_points = np.where(self.crystal > 0.5)
        if len(crystal_points[0]) > 0:
            r = np.sqrt((crystal_points[0] - self.center)**2 + (crystal_points[1] - self.center)**2)
            self.max_radius = max(self.max_radius, r.max())
        
        # Apply melting/smoothing at near-freezing temperatures
        if self.temperature > -2:
            # Identify isolated/sharp protrusions that would melt first
            crystal_eroded = ndimage.binary_erosion(self.crystal)
            melting_sites = np.logical_and(self.crystal, np.logical_not(crystal_eroded))
            
            # Apply probabilistic melting
            melt_probability = np.random.random(self.crystal.shape) * self.melting_threshold
            melt_mask = (melt_probability > 0.95) & melting_sites
            self.crystal[melt_mask] = 0
    
    def apply_symmetry(self):
        """Apply symmetry constraints to enforce the snowflake's crystallographic symmetry."""
        # Get the first sector
        sector_0 = (self.sector_map == 0)
        
        # Extract crystal data from sector 0
        sector_0_data = self.crystal * sector_0
        
        # Reset crystal outside of sector 0
        self.crystal = self.crystal * sector_0
        
        # Apply rotational symmetry to copy sector 0 to all sectors
        for i in range(1, self.symmetry_order):
            # Rotate sector 0 data to this sector
            rotated = ndimage.rotate(sector_0_data, i * 360 / self.symmetry_order, 
                                    reshape=False, order=1, mode='constant', cval=0)
            
            # Add to crystal
            self.crystal = np.maximum(self.crystal, rotated)
    
    def step(self):
        """Perform one time step of the simulation."""
        # Track step count
        self.step_count += 1
        
        # Apply symmetry if needed (every few steps for efficiency)
        if self.step_count % 5 == 0:
            self.apply_symmetry()
        
        # Update the tip bias map
        if self.step_count % 3 == 0:
            self.update_tip_bias_map()
        
        # Diffuse vapor around the crystal
        self.diffuse_vapor()
        
        # Grow the crystal based on local conditions
        self.grow_crystal()
        
        # Re-apply boundary conditions
        self.apply_boundary_conditions()
        
        # Every 20 steps, recalculate physical parameters in case of temperature changes
        if self.step_count % 20 == 0:
            self.compute_physical_parameters()
    
    def run(self, steps):
        """Run the simulation for a specified number of steps."""
        for _ in tqdm(range(steps), desc="Simulating snowflake growth"):
            self.step()
            
            # Stop if crystal reaches the boundary
            if self.max_radius > 0.48 * self.size * self.resolution_factor:
                print(f"Stopping: crystal reached boundary after {self.step_count} steps")
                break
    
    def get_display_crystal(self):
        """Get the crystal data downsampled for display if using higher resolution."""
        if self.resolution_factor > 1.0:
            # Downsample for display
            zoom_factor = 1.0 / self.resolution_factor
            return ndimage.zoom(self.crystal, zoom_factor, order=1)
        else:
            return self.crystal
    
    def get_display_vapor(self):
        """Get the vapor data downsampled for display if using higher resolution."""
        if self.resolution_factor > 1.0:
            # Downsample for display
            zoom_factor = 1.0 / self.resolution_factor
            return ndimage.zoom(self.vapor, zoom_factor, order=1)
        else:
            return self.vapor
    
  def visualize(self, include_vapor=True, cmap='ice', dpi=150, save=False, filename=None):
        """Visualize the current state of the snowflake simulation."""
        # Get display data
        crystal_display = self.get_display_crystal()
        
        if include_vapor:
            # Set up figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
            
            # Custom colormaps
            if cmap == 'ice':
                # Create ice-blue colormap for crystal
                ice_colors = [(0.9, 0.95, 1), (0.6, 0.8, 0.9), (0.2, 0.4, 0.8)]
                ice_cmap = LinearSegmentedColormap.from_list("ice", ice_colors)
                
                # Create vapor colormap
                vapor_colors = [(0, 0, 0.3), (0.1, 0.2, 0.5), (0.5, 0.5, 0.8)]
                vapor_cmap = LinearSegmentedColormap.from_list("vapor", vapor_colors)
            else:
                ice_cmap = plt.cm.winter
                vapor_cmap = plt.cm.cool
            
            # Plot crystal
            ax1.imshow(crystal_display, cmap=ice_cmap)
            ax1.set_title("Snowflake Crystal Structure")
            ax1.axis('off')
            
            # Plot vapor field
            vapor_display = self.get_display_vapor()
            vapor_img = ax2.imshow(vapor_display, cmap=vapor_cmap)
            ax2.set_title("Water Vapor Concentration")
            ax2.axis('off')
            
            # Add a colorbar for vapor concentration
            cbar = plt.colorbar(vapor_img, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Vapor Density')
            
        else:
            # Set up figure with just the crystal
            fig, ax1 = plt.subplots(figsize=(8, 8), dpi=dpi)
            
            # Create ice-blue colormap
            ice_colors = [(0.9, 0.95, 1), (0.6, 0.8, 0.9), (0.2, 0.4, 0.8)]
            ice_cmap = LinearSegmentedColormap.from_list("ice", ice_colors)
            
            # Plot crystal with black background
            ax1.set_facecolor('black')
            ax1.imshow(crystal_display, cmap=ice_cmap)
            
            # Add title with simulation parameters
            title = f"Snowflake Simulation: T={self.temperature}°C, RH={self.humidity*100:.0f}%"
            ax1.set_title(title, color='white')
            ax1.axis('off')
        
        # Add information about physical parameters
        param_text = (
            f"Temperature: {self.temperature}°C\n"
            f"Humidity: {self.humidity*100:.0f}%\n"
            f"Growth steps: {self.step_count}"
        )
        
        plt.figtext(0.01, 0.01, param_text, fontsize=8)
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save:
            if filename is None:
                # Create filename based on parameters
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"snowflake_T{self.temperature}_H{int(self.humidity*100)}_{timestamp}.png"
            
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to {filename}")
        
        return fig
    
    def create_animation(self, total_frames=100, interval=50, save_gif=False, filename=None):
        """Create an animation of the snowflake growth process."""
        # Create a copy of the current simulation state
        original_crystal = self.crystal.copy()
        original_vapor = self.vapor.copy()
        original_step_count = self.step_count
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create ice-blue colormap
        ice_colors = [(0.9, 0.95, 1), (0.6, 0.8, 0.9), (0.2, 0.4, 0.8)]
        ice_cmap = LinearSegmentedColormap.from_list("ice", ice_colors)
        
        # Set up black background
        ax.set_facecolor('black')
        
        # Initial plot
        crystal_display = self.get_display_crystal()
        img = ax.imshow(crystal_display, cmap=ice_cmap)
        
        # Add title
        title = ax.set_title(f"Snowflake Growth: T={self.temperature}°C, Step 0", color='white')
        ax.axis('off')
        
        # Update function for animation
        def update(frame):
            # Run the simulation for a few steps
            for _ in range(3):  # Multiple steps per frame for faster growth
                self.step()
            
            # Update the display
            crystal_display = self.get_display_crystal()
            img.set_array(crystal_display)
            
            # Update the title
            title.set_text(f"Snowflake Growth: T={self.temperature}°C, Step {self.step_count}")
            
            return [img, title]
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=True)
        
        # Save animation if requested
        if save_gif:
            if filename is None:
                # Create filename based on parameters
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"snowflake_animation_T{self.temperature}_H{int(self.humidity*100)}_{timestamp}.gif"
            
            ani.save(filename, writer='pillow', fps=15)
            print(f"Saved animation to {filename}")
        
        # Restore original state
        self.crystal = original_crystal
        self.vapor = original_vapor
        self.step_count = original_step_count
        
        return ani
    
    def create_comparison(temperatures=None, humidities=None, steps=80):
        """Create a comparison of snowflakes grown under different conditions."""
        if temperatures is None:
            temperatures = [-2, -5, -10, -15, -20]
        
        if humidities is None:
            humidities = [0.8, 0.95]
        
        # Grid dimensions
        rows = len(humidities)
        cols = len(temperatures)
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        # Flatten axes if only one row or column
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        # Create ice-blue colormap
        ice_colors = [(0.9, 0.95, 1), (0.6, 0.8, 0.9), (0.2, 0.4, 0.8)]
        ice_cmap = LinearSegmentedColormap.from_list("ice", ice_colors)
        
        # Create snowflakes for each condition
        for i, humidity in enumerate(humidities):
            for j, temp in enumerate(temperatures):
                # Create and run simulation
                sim = AdvancedSnowflakeSimulation(
                    size=200,  # Smaller size for comparison
                    temperature=temp,
                    humidity=humidity
                )
                sim.run(steps)
                
                # Set black background
                axes[i][j].set_facecolor('black')
                
                # Display crystal
                crystal_display = sim.get_display_crystal()
                axes[i][j].imshow(crystal_display, cmap=ice_cmap)
                
                # Set title
                axes[i][j].set_title(f"T={temp}°C, RH={humidity*100:.0f}%", color='white')
                axes[i][j].axis('off')
        
        plt.tight_layout()
        return fig


def main():
    """Main function to run the simulation with command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Snowflake Growth Simulation")
    
    # Add arguments
    parser.add_argument("--size", type=int, default=400, help="Size of the simulation grid")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--temp", type=float, default=-15, help="Temperature in Celsius")
    parser.add_argument("--humidity", type=float, default=0.95, help="Relative humidity (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--sym", type=int, default=6, help="Symmetry order (6 for normal snowflakes)")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--save", action="store_true", help="Save output")
    parser.add_argument("--vapor", action="store_true", help="Include vapor field in visualization")
    parser.add_argument("--compare", action="store_true", help="Create comparison across temperatures")
    
    args = parser.parse_args()
    
    # Create output directory if saving
    if args.save and not os.path.exists("snowflake_output"):
        os.makedirs("snowflake_output")
    
    # Run comparison if requested
    if args.compare:
        print("Creating temperature comparison...")
        fig = AdvancedSnowflakeSimulation.create_comparison(
            temperatures=[-2, -5, -10, -15, -20],
            humidities=[0.8, 0.95],
            steps=80
        )
        
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"snowflake_output/comparison_{timestamp}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {filename}")
        
        plt.show()
        return
    
    # Create simulation
