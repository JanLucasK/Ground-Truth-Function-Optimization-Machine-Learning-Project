import numpy as np
import pandas as pd
import torch
import bbobtorch
import matplotlib.pyplot as plt
from pyswarm import pso
import matplotlib.animation as animation


class PSO_optimizer():
    def __init__(self, input_bounds=None, swarmsize=50) -> None:   
        self.input_bounds = input_bounds
        self.swarmsize = swarmsize
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
    
    def optimize(self, model_path, function='f_01', niter=100, swarmsize=50, image_name = "Default", save_image=False, seed=42):
        self.model = self.load_model(model_path)
        self.model.eval()
        if function == "f_01":
            self.bbob = bbobtorch.create_f01(2, seed=42)
        elif function=="f_03":
            self.bbob = bbobtorch.create_f03(2, seed=42)
        else:
            self.bbob = bbobtorch.create_f24(2, seed=42)
        
        self.bbob_path = pd.DataFrame(columns=["x1", "x2", "y"])
        self.model_path = pd.DataFrame(columns=["x1", "x2", "y"])
        self.path = []

        self.save_image = save_image
        self.image_name = image_name

        # Use PSO optimizer
        result_nn, _ = pso(self.call_nn, swarmsize=swarmsize, maxiter=niter, minstep=1e-5, lb = [-5,-5], ub=[5,5])
        model_path = np.array(self.path)
        self.path = []

        result_bbob, _ = pso(self.call_bbob, swarmsize=swarmsize, maxiter=niter, minstep=1e-5, lb = [-5,-5], ub=[5,5])
        bbob_path = np.array(self.path)
        
        # fig = self.visualize_paths(result_bbob, result_nn)
        fig = self.visualize_paths(self.bbob_path, self.model_path, result_bbob, result_nn, swarmsize)

        return result_nn, result_bbob, fig
    
    
    def visualize_paths(self, path_bbob, path_model, result_bbob, result_model, swarmsize):
        x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))

        # Convert the scaled grid data to PyTorch tensor
        grid_data_tensor = torch.tensor(np.column_stack((x_grid.ravel(), y_grid.ravel())), dtype=torch.float32)

        # Set the model to evaluation mode
        self.model.eval()

        # Pass the grid data through the model to get predictions
        with torch.no_grad():
            predictions = self.model(grid_data_tensor).numpy().reshape(x_grid.shape)

        function_values_bbob = self._get_gt_function(x_grid, y_grid)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        ax1 = axes[0]  # Define ax1 for the left plot prediction
        ax2 = axes[1]  # Define ax2 for the right plot bbob

        ax1.contourf(x_grid, y_grid, predictions, levels=100, cmap='viridis')
        ax2.contourf(x_grid, y_grid, function_values_bbob, levels=100, cmap='viridis')

        for ax, path in zip([ax1, ax2], [path_model, path_bbob]):
            for j in range(0, len(path), swarmsize):
                best_particle = path[j:j+swarmsize][np.argmin(path[j:j+swarmsize][:, 2])]
                ax.scatter(path[j:j+swarmsize, 0], path[j:j+swarmsize, 1], c='red', s=10, marker='o', alpha=0.5)
                ax.scatter(best_particle[0], best_particle[1], c='green', s=100, marker='o')  # Mark gBest with a cross

            if ax is ax1:  # Plot result for the left subplot
                ax.scatter(x=result_model[0], y=result_model[1], c='black', s=250, marker='x')
            else:  # Plot result for the right subplot
                ax.scatter(x=result_bbob[0], y=result_bbob[1], c='black', s=250, marker='x')

        ax1.set_title("Model Predictions")
        ax2.set_title("BBOB Function")

        plt.tight_layout()
        if self.save_image:
            plt.savefig('24_swarm_iterations.png')  # Save the plot as a PNG image
        
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
        x_tensor = x_tensor.view(1,-1)
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

