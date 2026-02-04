import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from scipy.ndimage import convolve, rotate

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
        # Apply diffusion using a convolution-like approach
        kernel = np.array([[0.05, 0.2, 0.05], 
                           [0.2, 0, 0.2], 
                           [0.05, 0.2, 0.05]])
        
        # Only diffuse in non-crystallized areas
        non_crystal = self.grid <= self.threshold
        diffusion = convolve(self.grid, kernel, mode='constant', cval=0)

        # Growth and noise
        growth = diffusion * self.growth_rate
        noise = np.random.randn(*self.grid.shape) * self.noise_level

        # Update the grid
        update = (growth + noise) * non_crystal
        self.grid = np.clip(self.grid + update, 0, 1)
        
        # Apply hexagonal symmetry if enabled
        if self.hexagonal:
            # We use a simple rotational averaging for symmetry
            symmetric_grid = np.copy(self.grid)
            for i in range(1, 6):
                rotated = rotate(self.grid, i * 60, reshape=False, order=1, mode='constant', cval=0)
                symmetric_grid = np.maximum(symmetric_grid, rotated)
            self.grid = symmetric_grid
        
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
    # Check for ipywidgets for Jupyter environments
    try:
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
    except ImportError:
        matplotlib_interactive_simulation()

def matplotlib_interactive_simulation():
    """Run an interactive simulation using matplotlib widgets."""
    size = 200
    sim = SnowflakeSimulation(size=size)

    fig, ax = plt.subplots(figsize=(10, 11))
    plt.subplots_adjust(bottom=0.25)

    # Create colormap
    colors = [(0, 0, 0.5), (0, 0.5, 1), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("snowflake", colors)

    img = ax.imshow(sim.grid, cmap=cmap, origin='lower')
    ax.set_title("Turing Diffusion Snowflake Growth")
    ax.axis('off')

    # Define axes for sliders
    ax_diff = plt.axes([0.2, 0.15, 0.65, 0.03])
    ax_growth = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_noise = plt.axes([0.2, 0.05, 0.65, 0.03])

    # Create sliders
    s_diff = Slider(ax_diff, 'Diffusion', 0.1, 2.0, valinit=sim.diffusion_rate)
    s_growth = Slider(ax_growth, 'Growth', 0.001, 0.05, valinit=sim.growth_rate)
    s_noise = Slider(ax_noise, 'Noise', 0.0, 0.2, valinit=sim.noise_level)

    def update_params(val):
        sim.diffusion_rate = s_diff.val
        sim.growth_rate = s_growth.val
        sim.noise_level = s_noise.val

    s_diff.on_changed(update_params)
    s_growth.on_changed(update_params)
    s_noise.on_changed(update_params)

    # Add buttons
    ax_reset = plt.axes([0.8, 0.01, 0.1, 0.04])
    ax_step = plt.axes([0.65, 0.01, 0.1, 0.04])
    ax_run = plt.axes([0.5, 0.01, 0.1, 0.04])

    btn_reset = Button(ax_reset, 'Reset')
    btn_step = Button(ax_step, 'Step')
    btn_run = Button(ax_run, 'Run 10')

    def reset(event):
        sim.__init__(size=size, diffusion_rate=s_diff.val,
                    growth_rate=s_growth.val, noise_level=s_noise.val)
        img.set_array(sim.grid)
        fig.canvas.draw_idle()

    def step(event):
        sim.step()
        img.set_array(sim.grid)
        fig.canvas.draw_idle()

    def run_10(event):
        for _ in range(10):
            sim.step()
        img.set_array(sim.grid)
        fig.canvas.draw_idle()

    btn_reset.on_clicked(reset)
    btn_step.on_clicked(step)
    btn_run.on_clicked(run_10)

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
    
    # Option 4: Interactive simulation
    # print("Running interactive simulation...")
    # interactive_simulation()
    
    print("Done!")
