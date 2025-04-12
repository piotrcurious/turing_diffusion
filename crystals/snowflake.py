import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

class SnowflakeSimulation:
    def __init__(self, size=300, hexagonal=True, diffusion_rate=1.0, 
                 growth_rate=0.015, noise_level=0.1, threshold=0.7):
        """
        Initialize the snowflake simulation.
        
        Parameters:
        -----------
        size : int
            Size of the grid
        hexagonal : bool
            Whether to enforce hexagonal symmetry
        diffusion_rate : float
            Rate of diffusion in the system
        growth_rate : float
            Rate of crystal growth
        noise_level : float
            Level of random noise in the system
        threshold : float
            Threshold for crystal formation
        """
        self.size = size
        self.hexagonal = hexagonal
        self.diffusion_rate = diffusion_rate
        self.growth_rate = growth_rate
        self.noise_level = noise_level
        self.threshold = threshold
        
        # Center of the grid
        self.center = size // 2
        
        # Initialize grid with zeros
        self.grid = np.zeros((size, size))
        
        # Seed the center point
        self.grid[self.center, self.center] = 1.0
        
        # Initialize the mask for hexagonal symmetry
        if hexagonal:
            self.symmetry_points = 6
            self.mask = self._create_hexagonal_mask()
        else:
            self.symmetry_points = 1
            self.mask = None
    
    def _create_hexagonal_mask(self):
        """Create a mask for hexagonal symmetry."""
        x = np.arange(self.size) - self.center
        y = np.arange(self.size) - self.center
        xx, yy = np.meshgrid(x, y)
        
        # Calculate angle from center
        angle = np.arctan2(yy, xx)
        
        # Discretize angle into 6 segments
        sector = np.round(angle * 3 / np.pi) % 6
        
        return sector
    
    def _apply_hexagonal_symmetry(self, update):
        """Apply hexagonal symmetry to the update."""
        symmetric_update = np.zeros_like(update)
        
        # For each sector, rotate the updates to create 6-fold symmetry
        for i in range(6):
            # Get the points in this sector
            sector_points = (self.mask == 0)
            
            # Rotate the update matrix to each sector
            rotated_update = np.zeros_like(update)
            rotated_update[sector_points] = update[sector_points]
            
            # Rotate the matrix by 60 degrees (π/3 radians)
            symmetric_update += rotated_update
        
        return symmetric_update / 6.0
    
    def step(self):
        """Perform one step of the simulation."""
        # Create a copy of the current grid
        new_grid = self.grid.copy()
        
        # Apply diffusion using a convolution-like approach
        kernel = np.array([[0.05, 0.2, 0.05], 
                           [0.2, 0, 0.2], 
                           [0.05, 0.2, 0.05]])
        
        # Manual convolution
        for i in range(1, self.size-1):
            for j in range(1, self.size-1):
                # Skip if this point is already crystallized
                if self.grid[i, j] > self.threshold:
                    continue
                
                diffusion = 0
                for ki in range(-1, 2):
                    for kj in range(-1, 2):
                        if ki == 0 and kj == 0:
                            continue
                        diffusion += self.grid[i+ki, j+kj] * kernel[ki+1, kj+1]
                
                # Calculate growth based on diffusion
                growth = diffusion * self.growth_rate
                
                # Add some random noise
                noise = np.random.randn() * self.noise_level
                
                # Update the grid
                new_grid[i, j] += growth + noise
                
                # Threshold to ensure values stay in reasonable range
                new_grid[i, j] = np.clip(new_grid[i, j], 0, 1)
        
        # Apply hexagonal symmetry if enabled
        if self.hexagonal:
            # Calculate the update
            update = new_grid - self.grid
            
            # Calculate distance from center
            x = np.arange(self.size) - self.center
            y = np.arange(self.size) - self.center
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)
            
            # Calculate angle from center
            angle = np.arctan2(yy, xx)
            
            # Get the first sector (0 to π/3)
            first_sector = (angle >= 0) & (angle < np.pi/3)
            
            # Apply update only to first sector
            sector_update = np.zeros_like(update)
            sector_update[first_sector] = update[first_sector]
            
            # Apply 6-fold symmetry
            for i in range(6):
                rot_angle = i * np.pi/3
                rot_cos = np.cos(rot_angle)
                rot_sin = np.sin(rot_angle)
                
                # Rotate coordinates
                xr = xx * rot_cos - yy * rot_sin
                yr = xx * rot_sin + yy * rot_cos
                
                # Map back to pixel coordinates
                xi = np.round(xr + self.center).astype(int)
                yi = np.round(yr + self.center).astype(int)
                
                # Filter valid indices
                valid = (xi >= 0) & (xi < self.size) & (yi >= 0) & (yi < self.size)
                
                # Apply update to rotated positions
                for x, y in zip(xi[first_sector & valid].flat, yi[first_sector & valid].flat):
                    if 0 <= x < self.size and 0 <= y < self.size:
                        new_grid[y, x] = self.grid[y, x] + sector_update[first_sector][0]
        
        self.grid = new_grid
        
        # Apply threshold
        self.grid[self.grid > self.threshold] = 1.0
    
    def run(self, steps=100):
        """Run the simulation for a given number of steps."""
        for _ in range(steps):
            self.step()
    
    def plot(self, ax=None):
        """Plot the current state of the simulation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a cool blue colormap for snowflakes
        colors = [(0, 0, 0.5), (0, 0.5, 1), (1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("snowflake", colors)
        
        ax.imshow(self.grid, cmap=cmap, origin='lower')
        ax.set_title("Turing Diffusion Snowflake Simulation")
        ax.axis('off')
        
        return ax

def animate_snowflake_growth(steps=100, interval=100, save_gif=False):
    """
    Animate the snowflake growth process.
    
    Parameters:
    -----------
    steps : int
        Number of steps to simulate
    interval : int
        Interval between frames in milliseconds
    save_gif : bool
        Whether to save the animation as a GIF
    """
    # Create the simulation
    sim = SnowflakeSimulation(size=200, diffusion_rate=1.0, 
                            growth_rate=0.02, noise_level=0.005)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a cool blue colormap for snowflakes
    colors = [(0, 0, 0.5), (0, 0.5, 1), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("snowflake", colors)
    
    # Initialize the plot
    img = ax.imshow(sim.grid, cmap=cmap, origin='lower')
    ax.set_title("Turing Diffusion Snowflake Growth")
    ax.axis('off')
    
    def update(frame):
        # Update the simulation
        sim.step()
        
        # Update the plot
        img.set_array(sim.grid)
        
        return [img]
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    
    # Save the animation as a GIF if requested
    if save_gif:
        ani.save('snowflake_growth.gif', writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def compare_parameters():
    """Compare different parameter settings for snowflake growth."""
    # Different parameter settings to compare
    params = [
        {"diffusion_rate": 0.5, "growth_rate": 0.01, "noise_level": 0.01, "title": "Slow Growth, Low Noise"},
        {"diffusion_rate": 1.0, "growth_rate": 0.02, "noise_level": 0.05, "title": "Medium Growth, Medium Noise"},
        {"diffusion_rate": 2.0, "growth_rate": 0.03, "noise_level": 0.1, "title": "Fast Growth, High Noise"}
    ]
    
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Run simulations with different parameters
    for i, p in enumerate(params):
        sim = SnowflakeSimulation(size=200, diffusion_rate=p["diffusion_rate"], 
                                growth_rate=p["growth_rate"], noise_level=p["noise_level"])
        sim.run(steps=50)
        sim.plot(ax=axes[i])
        axes[i].set_title(p["title"])
    
    plt.tight_layout()
    plt.show()

def interactive_simulation():
    """Run an interactive simulation where the user can adjust parameters."""
    from ipywidgets import interact, FloatSlider
    
    @interact(
        diffusion_rate=FloatSlider(min=0.1, max=2.0, step=0.1, value=1.0),
        growth_rate=FloatSlider(min=0.001, max=0.05, step=0.001, value=0.015),
        noise_level=FloatSlider(min=0.0, max=0.2, step=0.01, value=0.05),
        steps=FloatSlider(min=10, max=200, step=10, value=50)
    )
    def run_sim(diffusion_rate, growth_rate, noise_level, steps):
        sim = SnowflakeSimulation(size=200, diffusion_rate=diffusion_rate, 
                                growth_rate=growth_rate, noise_level=noise_level)
        sim.run(steps=int(steps))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sim.plot(ax=ax)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Demonstrate different ways to use the simulation
    
    # Option 1: Simple static simulation
    print("Running static simulation...")
    sim = SnowflakeSimulation(size=200)
    sim.run(steps=50)
    sim.plot()
    plt.savefig('snowflake_static.png')
    plt.close()
    
    # Option 2: Animated growth
    print("Running animated simulation (this may take a moment)...")
    animate_snowflake_growth(steps=60, interval=100, save_gif=True)
    
    # Option 3: Compare different parameters
    print("Comparing different parameters...")
    compare_parameters()
    
    # Note: Uncomment to run interactive simulation in Jupyter notebook
    # print("Running interactive simulation...")
    # interactive_simulation()
    
    print("Done!")
