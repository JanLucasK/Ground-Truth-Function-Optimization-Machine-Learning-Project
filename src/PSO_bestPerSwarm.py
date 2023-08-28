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
        best_positions_bbob = []
        best_positions_model = []

        for i in range(niter):
            result_nn, _ = pso(self.call_nn, swarmsize=num_particles, maxiter=1, minstep=1e-5, lb=[-5, -5], ub=[5, 5])
            result_bbob, _ = pso(self.call_bbob, swarmsize=num_particles, maxiter=1, minstep=1e-5, lb=[-5, -5], ub=[5, 5])

            best_positions_model.append(result_nn)
            best_positions_bbob.append(result_bbob)

        fig = self.visualize_paths(best_positions_bbob, best_positions_model)

        return fig
    
    def visualize_paths(self, best_positions_bbob, best_positions_model):
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
        ax1 = axes[0]
        ax2 = axes[1]

        ax1.contourf(x_grid, y_grid, predictions, levels=100, cmap='viridis')
        ax2.contourf(x_grid, y_grid, function_values, levels=100, cmap='viridis')

        best_model_positions = np.array(best_positions_model)
        best_bbob_positions = np.array(best_positions_bbob)

        ax1.scatter(best_model_positions[:, 0], best_model_positions[:, 1], c='red', s=100, marker='o', alpha=0.5)
        ax2.scatter(best_bbob_positions[:, 0], best_bbob_positions[:, 1], c='blue', s=100, marker='o', alpha=0.5)

        for i in range(1, len(best_model_positions)):
            ax1.plot([best_model_positions[i-1, 0], best_model_positions[i, 0]],
                     [best_model_positions[i-1, 1], best_model_positions[i, 1]], c='red')

        for i in range(1, len(best_bbob_positions)):
            ax2.plot([best_bbob_positions[i-1, 0], best_bbob_positions[i, 0]],
                     [best_bbob_positions[i-1, 1], best_bbob_positions[i, 1]], c='blue')

        ax1.set_title('Best Particle Trajectory (Model)')
        ax2.set_title('Best Particle Trajectory (BBOB)')

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax2.set_xlim(-5, 5)
        ax2.set_ylim(-5, 5)

        plt.tight_layout()
        plt.savefig('24best_particle_trajectory.png')
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
model_path = 'models/v1/training_v1_f24_3.pth'
input_bounds = input_bounds=[(-5.0, 5.0), (-5.0, 5.0)]

optimizer = PSO_optimizer(model_path, input_bounds)
initial_guess = [0.0, 0.0]
niter = 20  # Number of iterations
num_particles = 50 # Number of particles each swarm

result_nn, result_bbob, fig = optimizer.optimize(initial_guess, niter)