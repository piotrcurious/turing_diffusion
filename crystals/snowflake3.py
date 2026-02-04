import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage as ndimage
from numba import jit
import time
import os
from datetime import datetime

class AdvancedSnowflakeModel:
    def __init__(self, 
                 size=400, 
                 symmetry_order=6,
                 vapor_diffusion_rate=0.95,
                 heat_diffusion_rate=1.2,
                 attachment_coefficient=0.5,
                 surface_tension=0.05,
                 temperature=-15.0,
                 humidity=0.85,
                 initial_seed_size=3,
                 boundary_vapor_density=1.0,
                 anisotropy_strength=0.3,
                 random_seed=None,
                 mode='dendrite'):
        """
        Advanced physical model for snowflake crystal growth based on Turing diffusion principles.
        
        Parameters:
        -----------
        size : int
            Size of the simulation grid (pixels)
        symmetry_order : int
            Order of symmetry (typically 6 for snowflakes)
        vapor_diffusion_rate : float
            Diffusion coefficient for water vapor
        heat_diffusion_rate : float
            Diffusion coefficient for thermal energy
        attachment_coefficient : float
            Probability coefficient for vapor attachment to crystal
        surface_tension : float
            Surface tension effect (controls smoothness)
        temperature : float
            Ambient temperature in Celsius (affects growth regime)
        humidity : float
            Relative humidity (affects vapor availability)
        initial_seed_size : int
            Size of the initial crystal seed
        boundary_vapor_density : float
            Vapor density at the boundary
        anisotropy_strength : float
            Strength of crystalline anisotropy (directionality of growth)
        random_seed : int or None
            Seed for random number generator
        mode : str
            Growth mode ('dendrite', 'plate', 'sector', 'fernlike', 'needle')
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Store parameters
        self.size = size
        self.symmetry_order = symmetry_order
        self.vapor_diffusion_rate = vapor_diffusion_rate
        self.heat_diffusion_rate = heat_diffusion_rate
        self.attachment_coefficient = attachment_coefficient
        self.surface_tension = surface_tension
        self.temperature = temperature
        self.humidity = humidity
        self.boundary_vapor_density = boundary_vapor_density
        self.anisotropy_strength = anisotropy_strength
        self.mode = mode
        
        # Center of the grid
        self.center = size // 2
        
        # Initialize phase field (1.0 = ice, 0.0 = vapor)
        self.phase = np.zeros((size, size))
        
        # Create initial seed at the center
        self._create_initial_seed(initial_seed_size)
        
        # Initialize vapor field (proportional to supersaturation)
        self.vapor = np.ones((size, size)) * humidity
        
        # Initialize temperature field (0 = ambient temperature)
        self.temperature_field = np.zeros((size, size))
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Initialize angular preference field based on mode
        self._initialize_growth_mode()
        
        # Cache for neighborhood calculations
        self._neighbors_cache = {}
        
        # Track iteration count
        self.iteration = 0
        
        # Prepare output directory for saving results
        self.output_dir = f"snowflake_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _initialize_growth_mode(self):
        """Initialize growth parameters based on selected mode."""
        # Create directional preference field
        x = np.arange(self.size) - self.center
        y = np.arange(self.size) - self.center
        xx, yy = np.meshgrid(x, y)
        
        # Calculate angle from center for each cell
        self.angle_field = np.arctan2(yy, xx)
        
        # Each growth mode has different angular preferences and parameters
        if self.mode == 'dendrite':
            # Sharp branching with secondary branches
            self.basal_preference = 0.8
            self.prism_preference = 0.2
            self.secondary_nucleation_rate = 0.03
            self.tip_growth_factor = 1.5
            self.edge_effects = 1.2
            
        elif self.mode == 'plate':
            # Thin, flat plate with minimal branching
            self.basal_preference = 0.9
            self.prism_preference = 0.1
            self.secondary_nucleation_rate = 0.01
            self.tip_growth_factor = 1.0
            self.edge_effects = 0.8
            
        elif self.mode == 'sector':
            # Broad, sectored plate
            self.basal_preference = 0.7
            self.prism_preference = 0.3
            self.secondary_nucleation_rate = 0.02
            self.tip_growth_factor = 1.1
            self.edge_effects = 1.0
            
        elif self.mode == 'fernlike':
            # Highly branched, fern-like structures
            self.basal_preference = 0.5
            self.prism_preference = 0.5
            self.secondary_nucleation_rate = 0.05
            self.tip_growth_factor = 1.8
            self.edge_effects = 1.5
            
        elif self.mode == 'needle':
            # Elongated prism growth
            self.basal_preference = 0.1
            self.prism_preference = 0.9
            self.secondary_nucleation_rate = 0.01
            self.tip_growth_factor = 2.0
            self.edge_effects = 0.7
        
        # Create anisotropy field based on crystalline structure
        # For hexagonal crystals, we create a 6-fold preference
        self.anisotropy_field = np.zeros((self.size, self.size))
        
        # Calculate radial distance from center
        self.radius = np.sqrt(xx**2 + yy**2)
        
        # Calculate anisotropy based on angle and mode
        for i in range(self.symmetry_order):
            angle = i * 2 * np.pi / self.symmetry_order
            self.anisotropy_field += np.cos(self.symmetry_order * (self.angle_field - angle))**2
            
        # Normalize
        self.anisotropy_field = self.anisotropy_field / self.anisotropy_field.max()
        
    def _create_initial_seed(self, seed_size):
        """Create the initial crystal seed."""
        # For a more realistic seed, we'll create a small hexagon
        x = np.arange(self.size) - self.center
        y = np.arange(self.size) - self.center
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        
        # Basic circular seed
        self.phase[r < seed_size] = 1.0
        
        # Add some hexagonal character
        angle = np.arctan2(yy, xx)
        for i in range(self.symmetry_order):
            a = i * 2 * np.pi / self.symmetry_order
            # Add slight extensions in the crystallographic directions
            extension = np.cos(self.symmetry_order * (angle - a))**2
            extended_radius = seed_size * (1 + 0.2 * extension)
            self.phase[r < extended_radius] = 1.0
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to the fields."""
        # Set vapor density at boundaries to maintain supersaturation
        margin = 5
        
        # Top and bottom boundaries
        self.vapor[:margin, :] = self.boundary_vapor_density
        self.vapor[-margin:, :] = self.boundary_vapor_density
        
        # Left and right boundaries
        self.vapor[:, :margin] = self.boundary_vapor_density
        self.vapor[:, -margin:] = self.boundary_vapor_density
        
        # Temperature is fixed at boundaries to ambient
        self.temperature_field[:margin, :] = 0
        self.temperature_field[-margin:, :] = 0
        self.temperature_field[:, :margin] = 0
        self.temperature_field[:, -margin:] = 0
        
    @jit(nopython=True)
    def _diffuse_field(self, field, diffusion_rate):
        """Apply diffusion to a field using the Laplace operator."""
        # Optimized diffusion calculation
        laplacian = (
            field[:-2, 1:-1] + field[2:, 1:-1] + 
            field[1:-1, :-2] + field[1:-1, 2:] - 
            4 * field[1:-1, 1:-1]
        )
        
        field_new = field.copy()
        field_new[1:-1, 1:-1] += diffusion_rate * laplacian
        
        return field_new
    
    def _calculate_growth_rate(self):
        """Calculate growth rate based on local conditions."""
        # Initialize growth rate field
        growth_rate = np.zeros_like(self.phase)
        
        # Calculate gradient of phase field (identifies interface)
        grad_x = np.zeros_like(self.phase)
        grad_y = np.zeros_like(self.phase)
        
        grad_x[:, 1:-1] = self.phase[:, 2:] - self.phase[:, :-2]
        grad_y[1:-1, :] = self.phase[2:, :] - self.phase[:-2, :]
        
        # Magnitude of gradient (interface region has non-zero gradient)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identify interface region (where gradient is significant)
        interface = (grad_mag > 0.01) & (self.phase < 0.9) & (self.phase > 0.1)
        
        # Calculate curvature (dividing by grad_mag where it's not too small)
        safe_divisor = np.maximum(grad_mag, 1e-10)
        curv_x = np.gradient(grad_x / safe_divisor, axis=1)
        curv_y = np.gradient(grad_y / safe_divisor, axis=0)
        curvature = curv_x + curv_y
        
        # Calculate directional preference based on crystal structure
        directional_factor = 1.0 + self.anisotropy_strength * (self.anisotropy_field - 0.5)
        
        # Distance from center affects growth (normalized)
        distance_factor = np.minimum(self.radius / (self.size/2 * 0.7), 1.0)
        tip_enhancement = 1.0 + (self.tip_growth_factor - 1.0) * distance_factor
        
        # At the interface, growth depends on:
        # 1. Local vapor concentration (diffusion limited)
        # 2. Temperature (affects attachment probability)
        # 3. Surface tension (curvature effect)
        # 4. Crystalline anisotropy
        
        temperature_factor = 1.0 - 0.02 * self.temperature_field
        
        # Calculate actual growth rate at interface
        growth_rate[interface] = (
            # Vapor contribution
            self.attachment_coefficient * self.vapor[interface] * 
            # Temperature effect
            temperature_factor[interface] *
            # Surface tension effect (curvature dependent)
            (1.0 - self.surface_tension * curvature[interface]) * 
            # Directional preference from crystal structure
            directional_factor[interface] *
            # Tip enhancement effect
            tip_enhancement[interface]
        )
        
        # Ensure growth rate is non-negative
        growth_rate = np.maximum(growth_rate, 0)
        
        # Apply symmetry
        growth_rate = self._apply_symmetry(growth_rate)
        
        return growth_rate
    
    def _apply_symmetry(self, field):
        """Apply n-fold symmetry to a field."""
        # Get the first sector
        x = np.arange(self.size) - self.center
        y = np.arange(self.size) - self.center
        xx, yy = np.meshgrid(x, y)
        
        # Calculate angle and radius for each point
        angle = np.arctan2(yy, xx)
        r = np.sqrt(xx**2 + yy**2)
        
        # Define the first sector (from 0 to 2π/n)
        sector_angle = 2 * np.pi / self.symmetry_order
        first_sector = (angle >= 0) & (angle < sector_angle)
        
        # Store the values from the first sector
        sector_values = field.copy()
        sector_values[~first_sector] = 0
        
        # Create symmetric field
        symmetric_field = np.zeros_like(field)
        
        # Apply rotational symmetry
        for i in range(self.symmetry_order):
            # Rotation angle
            rot_angle = i * sector_angle
            
            # Rotation matrix elements
            cos_rot = np.cos(rot_angle)
            sin_rot = np.sin(rot_angle)
            
            # Rotate coordinates
            x_rot = xx * cos_rot - yy * sin_rot
            y_rot = xx * sin_rot + yy * cos_rot
            
            # Convert to indices
            x_idx = np.round(x_rot + self.center).astype(int)
            y_idx = np.round(y_rot + self.center).astype(int)
            
            # Filter valid indices
            valid = (x_idx >= 0) & (x_idx < self.size) & (y_idx >= 0) & (y_idx < self.size)
            
            # Map values from first sector to rotated positions
            for j in range(self.size):
                for k in range(self.size):
                    if first_sector[j, k] and valid[j, k]:
                        x_dest, y_dest = x_idx[j, k], y_idx[j, k]
                        symmetric_field[y_dest, x_dest] = sector_values[j, k]
        
        return symmetric_field
    
    def step(self):
        """Perform one step of the simulation."""
        # 1. Diffuse vapor field
        self.vapor = self._diffuse_field(self.vapor, self.vapor_diffusion_rate)
        
        # 2. Diffuse temperature field
        self.temperature_field = self._diffuse_field(self.temperature_field, self.heat_diffusion_rate)
        
        # 3. Calculate growth rate
        growth_rate = self._calculate_growth_rate()
        
        # 4. Update phase field based on growth rate
        self.phase += growth_rate
        
        # Clip phase field to [0, 1]
        self.phase = np.clip(self.phase, 0, 1)
        
        # 5. Update vapor field (depletion due to crystal growth)
        self.vapor -= growth_rate
        
        # 6. Update temperature field (release latent heat during freezing)
        self.temperature_field += 0.5 * growth_rate
        
        # 7. Apply boundary conditions
        self._apply_boundary_conditions()
        
        # 8. Add random nucleation at tips for secondary branching
        self._add_secondary_branches()
        
        # Increment iteration counter
        self.iteration += 1
    
    def _add_secondary_branches(self):
        """Add secondary branching through controlled nucleation."""
        # Identify potential sites for secondary branching
        # These are typically at the edges of the crystal
        
        # Calculate phase gradient magnitude (edges have high gradient)
        grad_x = np.zeros_like(self.phase)
        grad_y = np.zeros_like(self.phase)
        
        grad_x[:, 1:-1] = self.phase[:, 2:] - self.phase[:, :-2]
        grad_y[1:-1, :] = self.phase[2:, :] - self.phase[:-2, :]
        
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # Consider points that:
        # 1. Are at crystal edges (high gradient)
        # 2. Have enough vapor for growth
        # 3. Are not too close to the center
        
        minimum_radius = self.size * 0.1
        potential_sites = (
            (edge_strength > 0.2) & 
            (self.vapor > 0.3) & 
            (self.radius > minimum_radius) &
            (self.phase > 0.3) & (self.phase < 0.7)
        )
        
        # Apply probability based on mode
        nucleation_probability = self.secondary_nucleation_rate
        
        # Generate random values
        random_field = np.random.random(self.phase.shape)
        
        # Determine nucleation sites
        nucleation_sites = potential_sites & (random_field < nucleation_probability)
        
        # Add small crystal seeds at nucleation sites
        nucleation_strength = 0.3  # Smaller than primary branches
        self.phase[nucleation_sites] += nucleation_strength
        
        # Apply symmetry to ensure consistent pattern
        self.phase = self._apply_symmetry(self.phase)
        
        # Clip to valid range
        self.phase = np.clip(self.phase, 0, 1)
    
    def run(self, steps=100, save_interval=None):
        """Run the simulation for a given number of steps."""
        # Create output directory if saving results
        if save_interval is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Track time
        start_time = time.time()
        
        for step in range(steps):
            self.step()
            
            # Save intermediate results if requested
            if save_interval is not None and step % save_interval == 0:
                self.save(f"{self.output_dir}/snowflake_{step:04d}.png")
            
            # Print progress
            if (step + 1) % max(1, steps // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step + 1}/{steps} completed. Elapsed time: {elapsed:.2f}s")
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    
    def save(self, filename):
        """Save the current state as an image."""
        plt.figure(figsize=(10, 10))
        self.plot()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot(self, ax=None, show_fields=False):
        """Plot the current state of the simulation."""
        if ax is None and not show_fields:
            fig, ax = plt.subplots(figsize=(10, 10))
        elif show_fields:
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            ax = axes[0, 0]
        
        # Create a cool blue colormap for snowflakes
        colors = [(0, 0, 0.2), (0, 0.5, 0.9), (0.9, 0.9, 1)]
        cmap_snow = LinearSegmentedColormap.from_list("snowflake", colors)
        
        # Plot the phase field
        im = ax.imshow(self.phase, cmap=cmap_snow, origin='lower', 
                       vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(f"Snowflake Growth - Iteration {self.iteration}\n"
                     f"T: {self.temperature}°C, Mode: {self.mode}")
        ax.axis('off')
        
        if show_fields:
            # Show vapor field
            vapor_ax = axes[0, 1]
            vapor_ax.imshow(self.vapor, cmap='Blues', origin='lower')
            vapor_ax.set_title("Vapor Field")
            vapor_ax.axis('off')
            
            # Show temperature field
            temp_ax = axes[1, 0]
            temp_ax.imshow(self.temperature_field, cmap='hot', origin='lower')
            temp_ax.set_title("Temperature Field")
            temp_ax.axis('off')
            
            # Show anisotropy field
            anis_ax = axes[1, 1]
            anis_ax.imshow(self.anisotropy_field, cmap='viridis', origin='lower')
            anis_ax.set_title("Anisotropy Field")
            anis_ax.axis('off')
            
            plt.tight_layout()
        
        return ax

def animate_snowflake_growth(model, steps=100, interval=50, save_gif=False):
    """
    Animate the snowflake growth process.
    
    Parameters:
    -----------
    model : AdvancedSnowflakeModel
        Initialized model to animate
    steps : int
        Number of steps to simulate
    interval : int
        Interval between frames in milliseconds
    save_gif : bool
        Whether to save the animation as a GIF
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a cool blue colormap for snowflakes
    colors = [(0, 0, 0.2), (0, 0.5, 0.9), (0.9, 0.9, 1)]
    cmap = LinearSegmentedColormap.from_list("snowflake", colors)
    
    # Initialize the plot
    img = ax.imshow(model.phase, cmap=cmap, origin='lower', 
                   vmin=0, vmax=1, interpolation='bilinear')
    title = ax.set_title(f"Snowflake Growth - Iteration {model.iteration}\n"
                         f"T: {model.temperature}°C, Mode: {model.mode}")
    ax.axis('off')
    
    def update(frame):
        # Update the simulation
        model.step()
        
        # Update the plot
        img.set_array(model.phase)
        title.set_text(f"Snowflake Growth - Iteration {model.iteration}\n"
                       f"T: {model.temperature}°C, Mode: {model.mode}")
        
        return [img, title]
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    
    # Save the animation as a GIF if requested
    if save_gif:
        filename = f"snowflake_{model.mode}_T{model.temperature}.gif"
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=15)
        print("Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def compare_temperature_effects():
    """Compare snowflake growth at different temperatures."""
    temperatures = [-5, -10, -15, -20]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, temp in enumerate(temperatures):
        # Create model with specific temperature
        model = AdvancedSnowflakeModel(size=300, temperature=temp, mode='dendrite')
        
        # Run simulation
        print(f"Simulating T = {temp}°C...")
        model.run(steps=50)
        
        # Plot result
        model.plot(ax=axes[i])
        axes[i].set_title(f"Temperature: {temp}°C")
    
    plt.tight_layout()
    plt.savefig("temperature_comparison.png", dpi=300)
    plt.show()

def compare_growth_modes():
    """Compare different snowflake growth modes."""
    modes = ['dendrite', 'plate', 'sector', 'fernlike', 'needle']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, mode in enumerate(modes):
        # Create model with specific mode
        model = AdvancedSnowflakeModel(size=300, temperature=-15, mode=mode)
        
        # Run simulation
        print(f"Simulating mode: {mode}...")
        model.run(steps=50)
        
        # Plot result
        model.plot(ax=axes[i])
        axes[i].set_title(f"Growth Mode: {mode}")
    
    # Hide the extra subplot
    if len(modes) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig("growth_mode_comparison.png", dpi=300)
    plt.show()

def moriyama_diagram():
    """Generate snowflakes across the Moriyama morphology diagram."""
    # Temperature and supersaturation pairs for different morphologies
    conditions = [
        {'temp': -2, 'humidity': 0.9, 'name': 'Thin Plate'},
        {'temp': -5, 'humidity': 0.8, 'name': 'Dendrite'},
        {'temp': -10, 'humidity': 0.7, 'name': 'Sector Plate'},
        {'temp': -15, 'humidity': 0.9, 'name': 'Fernlike Dendrite'},
        {'temp': -20, 'humidity': 0.8, 'name': 'Thin Plate'},
        {'temp': -25, 'humidity': 0.7, 'name': 'Hollow Column'}
    ]
    
    modes = ['plate', 'dendrite', 'sector', 'fernlike', 'plate', 'needle']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (condition, mode) in enumerate(zip(conditions, modes)):
        # Create model with specific conditions
        model = AdvancedSnowflakeModel(
            size=300, 
            temperature=condition['temp'], 
            humidity=condition['humidity'],
            mode=mode
        )
        
        # Run simulation
        print(f"Simulating {condition['name']}...")
        model.run(steps=60)
        
        # Plot result
        model.plot(ax=axes[i])
        axes[i].set_title(f"{condition['name']}\nT: {condition['temp']}°C, RH: {condition['humidity']*100:.0f}%")
    
    plt.tight_layout()
    plt.savefig("moriyama_diagram.png", dpi=300)
    plt.show()

def sample_3d_visualization(model):
    """Create a simple 3D visualization of the snowflake."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("3D plotting requires mpl_toolkits.mplot3d")
        return
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get grid coordinates
    x = np.arange(model.size)
    y = np.arange(model.size)
    X, Y = np.meshgrid(x, y)
    
    # Extract the phase field (crystal structure)
    Z = model.phase
    
    # Create 3D plot - use a height threshold to show only the crystal
    threshold = 0.5
    crystal_mask = Z > threshold
    
    # Thin out the points to avoid overloading the 3D renderer
    stride = 2
    X_thin = X[::stride, ::stride]
    Y_thin = Y[::stride, ::stride]
    Z_thin = Z[::stride, ::stride]
    mask_thin = crystal_mask[::stride, ::stride]
    
    # Plot only the crystal points
    ax.scatter(
        X_thin[mask_thin], 
        Y_thin[mask_thin], 
        Z_thin[mask_thin] * 0.1,  # Scale height for better visualization
        c=Z_thin[mask_thin],
        cmap='Blues',
        s=10,
        alpha=0.8
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Snowflake Crystal')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.2])
    
    plt.tight_layout()
    plt.savefig("snowflake_3d.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Run demonstration
    print("Advanced Snowflake Growth Simulation")
    print("------------------------------------")
    
    # Choose a growth mode
    # mode = 'dendrite'  # Options: 'dendrite', 'plate', 'sector', 'fernlike', 'needle'
    
    # Interactive model selection
    print("Available growth modes:")
    print("1. Dendrite (branched)")
    print("2. Plate (flat hexagonal)")
    print("3. Sector (segmented plate)")
    print("4. Fernlike (intricate branching)")
    print("5. Needle (columnar)")
    print("6. Compare all modes")
    print("7. Show temperature effects")
    print("8. Show Moriyama diagram (morphology map)")
    
    # Uncomment for interactive use
    # choice = input("Enter choice (1-8): ")
    
    # For demonstration, choose dendrite mode
    choice = "1"
    
    if choice == "6":
        compare_growth_modes()
    elif choice == "7":
        compare_temperature_effects()
    elif choice == "8":
        moriyama_diagram()
    else:
        # Map choice to mode
        mode_map = {
            "1": "dendrite",
            "2": "plate", 
            "3": "sector",
            "4": "fernlike",
            "5": "needle"
        }
        mode = mode_map.get(choice, "dendrite")
        
        print(f"Creating {mode} snowflake model...")
        model = AdvancedSnowflakeModel(
            size=300,
            mode=mode,
            temperature=-15,
            vapor_diffusion_rate=0.95,
            heat_diffusion_rate=1.2
        )
        
        # Run static simulation
        print("Running simulation...")
        model.run(steps=80)
        
        # Plot final result
        plt.figure(figsize=(12, 12))
