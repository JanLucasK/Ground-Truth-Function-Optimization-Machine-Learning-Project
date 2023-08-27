import numpy as np
import pandas as pd
import torch
import bbobtorch
import matplotlib.pyplot as plt
from pyswarm import pso
import matplotlib.animation as animation


class PSO_optimizer():
    def __init__(self, model_path, input_bounds=None) -> None:   
        self.model = self.load_model(model_path)
        self.input_bounds = input_bounds
        self.model.eval()
        self.bbob = bbobtorch.create_f24(2, seed=42)
        self.bbob_path = pd.DataFrame(columns=["x1", "x2", "y"])
        self.model_path = pd.DataFrame(columns=["x1", "x2", "y"])
        self.path = []
        self.particle_history = []
   
    def load_model(self, path):
       return torch.load(path)
   
   
    def call_nn(self, x):
        # Convert x to torch tensor
        if self.input_bounds:
            penalty = 0
            for val, (min_val, max_val) in zip(x, self.input_bounds):
                if val < min_val or val > max_val:
                    penalty += 1e5  # Adjust the penalty value based on your specific needs 
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # unsqueeze to add batch dimension
        y = self.model(x_tensor)
        y_scalar = torch.sum(y).item() + penalty
        
        model_point = [x[0], x[1], y_scalar]

        self.model_path.loc[len(self.model_path)] = model_point
        return y_scalar
    
    def optimize(self, initial_guess, niter=100):
        # Use PSO optimization instead of Basinhopping
        result_nn, _ = pso(self.call_nn, swarmsize=num_particles, maxiter=niter, minstep=1e-5, lb= [-5,-5], ub=[5, 5])
        result_bbob, _ = pso(self.call_bbob, swarmsize=num_particles, maxiter=niter, minstep=1e-5, lb= [-5,-5], ub=[5, 5])
        
        model_path = np.array(self.path)
        self.path = []

        # fig = self.visualize_paths(result_bbob, result_nn)
        fig = self.visualize_paths(self.bbob_path.values, self.model_path.values, result_bbob, result_nn)

        return result_nn, result_bbob, fig
    
    def visualize_paths(self, path_bbob, path_model, result_bbob, result_model):
        x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))

        # Convert the scaled grid data to PyTorch tensor
        grid_data_tensor = torch.tensor(np.column_stack((x_grid.ravel(), y_grid.ravel())), dtype=torch.float32)

        # Set the model to evaluation mode
        self.model.eval()

        # Pass the grid data through the model to get predictions
        with torch.no_grad():
            predictions = self.model(grid_data_tensor).numpy().reshape(x_grid.shape)

        function_values = self._get_gt_function(x_grid, y_grid)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        def animate(frame):
            axes[0].clear()
            axes[1].clear()

            ax1 = axes[0]
            ax2 = axes[1]

            ax1.contourf(x_grid, y_grid, predictions, levels=100, cmap='viridis')
            ax2.contourf(x_grid, y_grid, function_values, levels=100, cmap='viridis')

            for i, (ax, path) in enumerate(zip([ax1, ax2], [path_model, path_bbob])):
                start = frame * num_particles
                end = start + num_particles

                ax.scatter(path[start:end, 0], path[start:end, 1], c='red', s=10, marker='o', alpha=0.5)

                axes[0].scatter(x=result_model[0], y=result_model[1], c='black', s=250, marker='x')
                axes[1].scatter(x=result_bbob[0], y=result_bbob[1], c='black', s=250, marker='x')

        anim = animation.FuncAnimation(fig, animate, frames=len(path_model) // num_particles, interval=200, repeat=False)

        # Save the animation
        anim_filename = "/home/luka/Documents/24_pso_animation.gif"  # Specify the filename
        anim.save(anim_filename, writer='pillow', fps=2, dpi=150)

        plt.show()
        return plt
    
# call Bbob for minimization
    def call_bbob(self, x):
        if self.input_bounds:
            penalty = 0
            for val, (min_val, max_val) in zip(x, self.input_bounds):
                if val < min_val or val > max_val:
                    penalty += 1e5  # Adjust the penalty value based on your specific needs
        
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y = self.bbob(x_tensor)
        y_scalar = torch.sum(y).item() + penalty
        
        model_point = [x[0], x[1], y_scalar]

        self.bbob_path.loc[len(self.bbob_path)] = model_point
        return y_scalar
    
    def callback(self, x, f, accept):
        self.path.append(x)
    
    def _get_gt_function(self, x_grid, y_grid):
        fn = self.bbob
        flat_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Convert flat_grid to a PyTorch Tensor
        flat_grid_tensor = torch.tensor(flat_grid, dtype=torch.float32)

        # Evaluate the ground-truth function using the bbobtorch function
        results = fn(flat_grid_tensor)

        return results.numpy().reshape(x_grid.shape) 
    
# Create an instance of the PSO_optimizer class and run the optimization
model_path = 'models/training_v1_f24_3.pth'
input_bounds = input_bounds=[(-5.0, 5.0), (-5.0, 5.0)]

optimizer = PSO_optimizer(model_path, input_bounds)
initial_guess = [0.0, 0.0]
niter = 20  # Number of iterations
num_particles = 50 # Number of particles each swarm

result_nn, result_bbob, fig = optimizer.optimize(initial_guess, niter)