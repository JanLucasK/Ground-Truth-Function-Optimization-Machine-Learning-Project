import numpy as np
import pandas as pd
import torch
import bbobtorch
import matplotlib.pyplot as plt
from pyswarm import pso
import matplotlib.animation as animation


class PSO_optimizer():
    def __init__(self, input_bounds=None, swarmsize=50) -> None:
        # Initialize the PSO optimizer class
        self.input_bounds = input_bounds  # Input bounds for optimization
        self.swarmsize = swarmsize  # Number of particles in each swarm
        self.particle_history = []  # Store particle history
        self.bbob_path = pd.DataFrame(columns=["x1", "x2", "y"])  # DataFrame to store BBOB path
        self.model_path = pd.DataFrame(columns=["x1", "x2", "y"])  # DataFrame to store model path
        self.path = []  # Temporary path storage for callback

   
    def load_model(self, path):
       # Load a pre-trained PyTorch model from a given path
       return torch.load(path)
   
   
    def call_nn(self, x):
        # Call the neural network model and evaluate it for given input 'x'
        
        if self.input_bounds:
            penalty = 0
            for val, (min_val, max_val) in zip(x, self.input_bounds):
                if val < min_val or val > max_val:
                    penalty += 1e5  # Adjust the penalty value based on your specific needs 
        
        # Convert x to torch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # unsqueeze to add batch dimension
        y = self.model(x_tensor)
        y_scalar = torch.sum(y).item() + penalty
        
        model_point = [x[0], x[1], y_scalar] # Create a point with x, y, and scalar value

        self.model_path.loc[len(self.model_path)] = model_point # Add point to the model path
        return y_scalar
    
    def optimize(self, model_path, function='f_01', niter=100, swarmsize=50, image_name = "Default", save_image=False, seed=42):
        self.model = self.load_model(model_path)
        self.model.eval()
        # Create bbob functions
        if function == "f_01":
            self.bbob = bbobtorch.create_f01(2, seed=42)
        elif function=="f_03":
            self.bbob = bbobtorch.create_f03(2, seed=42)
        else:
            self.bbob = bbobtorch.create_f24(2, seed=42)

        self.save_image = save_image
        self.image_name = image_name

        # Use PSO optimizer
        result_nn, _ = pso(self.call_nn, swarmsize=swarmsize, maxiter=niter, minstep=1e-5, lb = [-5,-5], ub=[5,5])
        result_bbob, _ = pso(self.call_bbob, swarmsize=swarmsize, maxiter=niter, minstep=1e-5, lb = [-5,-5], ub=[5,5])
        
        model_path = np.array(self.path)
        self.path = []

        # Visualize the optimization paths
        fig = self.visualize_paths(self.bbob_path, self.model_path, result_bbob, result_nn, swarmsize)

        return result_nn, result_bbob, fig
    
    
    def visualize_paths(self, path_bbob, path_model, result_bbob, result_model, swarmsize):
        # Create a mesh grid for visualization
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

        # create contour plots
        ax1.contourf(x_grid, y_grid, predictions, levels=100, cmap='viridis')
        ax2.contourf(x_grid, y_grid, function_values_bbob, levels=100, cmap='viridis')

        for ax, path in zip([ax1, ax2], [path_model, path_bbob]):
            best_particles = []  # reset list 
            for j in range(0, len(path), swarmsize):
                swarm = path[j:j+swarmsize]
                best_particle = swarm.loc[swarm['y'].idxmin()]  # Find the best particle in the swarm
                best_particles.append(best_particle)

                ax.scatter(swarm['x1'], swarm['x2'], c='red', s=10, marker='o', alpha=0.5) # Plot swarm particles

            best_particles = pd.concat(best_particles)
            ax.scatter(best_particles['x1'], best_particles['x2'], c='green', s=100, marker='o') # Plot best particles

            if ax is ax1:
                ax.scatter(x=result_model[0], y=result_model[1], c='black', s=250, marker='x') # Plot model result
            else:
                ax.scatter(x=result_bbob[0], y=result_bbob[1], c='black', s=250, marker='x') # Plot BBOB result
             
        ax1.set_title("Model Predictions")
        ax2.set_title("BBOB Function")

        plt.tight_layout()
        if self.save_image:
            plt.savefig(f'images/PSO/50_SwarmSize_v3_5/{self.image_name}', dpi=300)  # Save the plot as a PNG image
        
        # plt.show()
        self.model_path = pd.DataFrame(columns=["x1", "x2", "y"])  # Reset data
        self.bbob_path = pd.DataFrame(columns=["x1", "x2", "y"])  # Reset data
        return plt
    
    def call_bbob(self, x):
        # call Bbob for minimization
        if self.input_bounds:
            penalty = 0
            for val, (min_val, max_val) in zip(x, self.input_bounds):
                if val < min_val or val > max_val:
                    penalty += 1e5  # Adjust the penalty value based on your specific needs
        
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor = x_tensor.view(1,-1) # Reshape the input tensor (because of f3)
        y = self.bbob(x_tensor)
        y_scalar = torch.sum(y).item() + penalty
        
        model_point = [x[0], x[1], y_scalar]

        self.bbob_path.loc[len(self.bbob_path)] = model_point  # Add the evaluated point to the BBOB path
        return y_scalar
    
    def callback(self, x, f, accept):
        # Callback function to record the path during optimization
        self.path.append(x)
    
    def _get_gt_function(self, x_grid, y_grid):
        fn = self.bbob # BBOB function for ground-truth evaluation
        flat_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Convert flat_grid to a PyTorch Tensor
        flat_grid_tensor = torch.tensor(flat_grid, dtype=torch.float32)

        # Evaluate the ground-truth function using the bbobtorch function
        results = fn(flat_grid_tensor)

        return results.numpy().reshape(x_grid.shape) 