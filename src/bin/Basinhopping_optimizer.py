import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import bbobtorch

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import basinhopping


class Basinhopping_optimizer():
    def __init__(self, model_path, input_bounds=None) -> None:   
        self.model = self.load_model(model_path)
        self.input_bounds = input_bounds
        self.model.eval()
        self.bbob = bbobtorch.create_f24(2, seed=42)
        self.bbob_path = pd.DataFrame(columns=["x1", "x2", "y"])
        self.model_path = pd.DataFrame(columns=["x1", "x2", "y"])
        self.path = []
   
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
    
    def optimize(self, initial_guess, niter=100, T=100, stepsize=0.1):
        result_nn = basinhopping(self.call_nn, initial_guess, niter=niter, T=T, stepsize=stepsize, callback=self.callback)
        model_path = np.array(self.path)
        self.path= []
        result_bbob = basinhopping(self.call_bbob, initial_guess, niter=niter, T=T, stepsize=stepsize, callback=self.callback)
        bbob_path = np.array(self.path)
        #self.visualize(result_nn, result_bbob)
        fig = self.visualize_paths(bbob_path, model_path, result_bbob, result_nn)

        return result_nn, result_bbob, fig
    
    def visualize_paths(self, path_bbob, path_model, result_bbob, result_model):
        x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))
        
        # Convert the scaled grid data to PyTorch tensor
        grid_data_tensor = torch.tensor(np.column_stack((x_grid.ravel(), y_grid.ravel())), dtype=torch.float32)

        # Set the model to evaluation mode
        self.model.eval()

        # Pass the grid data through the model to get predictions
        with torch.no_grad():
            #predictions = self.model(grid_data_tensor).view(x_grid.shape)
            predictions = self.model(grid_data_tensor).numpy().reshape(x_grid.shape)
            
        function_values = self._get_gt_function(x_grid, y_grid)
        
        fig, axes = plt.subplots(1, 2, figsize= (16,8))
        
        for ax, model_values, path, result in zip(axes, [predictions, function_values], [path_model, path_bbob], [result_model, result_bbob]):
            
            ax.contourf(x_grid, y_grid, model_values, levels=100, cmap='viridis')

            ax.plot(path[:,0], path[:,1], '-ro', markersize=1, linewidth=0.5)
            
            ax.scatter(x= result['x'][0], y= result['x'][1], c='black', s= 250, marker='x')
            
        plt.show(block='True')
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
    
    