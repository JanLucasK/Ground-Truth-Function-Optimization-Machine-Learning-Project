import numpy as np
import pandas as pd
import torch
import bbobtorch
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping

class Basinhopping_optimizer():
    #Class implements the optimization process with basin hopping both for 
    # the approximation from a neural net and the ground-truth function
    def __init__(self, input_bounds=None) -> None:   
        self.input_bounds = input_bounds

    def load_model(self, path):
        #Loads neural net model for optimization from filepath
        return torch.load(path)
   
   
    def call_nn(self, x):
        #Wrapper function to call the neural net for a point in the function landscape
        if self.input_bounds:
            penalty = 0
            for val, (min_val, max_val) in zip(x, self.input_bounds):
                if val < min_val or val > max_val:
                    penalty = 1e2  #Penalty to enforce search within function bounds
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x_tensor = x_tensor.view(1, -1) 
        y = self.model(x_tensor)
        y_scalar = torch.sum(y).item() + penalty #Function value as approximated by neural net + penalty for bounds
        
        model_point = [x[0], x[1], y_scalar]

        self.model_path.loc[len(self.model_path)] = model_point
        return y_scalar
    
    def call_bbob(self, x):
        #Wrapper function to call the BBOB-ground-truth for a point in the function landscape
        if self.input_bounds:
            penalty = 0
            for val, (min_val, max_val) in zip(x, self.input_bounds):
                if val < min_val or val > max_val:
                    penalty = 1e3
        
        x_tensor = torch.tensor([x], dtype=torch.float32)
        x_tensor = x_tensor.view(1, -1) 
        #print(self.bbob.dim, x_tensor)
        y = self.bbob(x_tensor)
        y_scalar = torch.sum(y).item() + penalty
        
        model_point = [x[0], x[1], y_scalar]
        self.bbob_path.loc[len(self.bbob_path)] = model_point
        return y_scalar
    
    def optimize(self, model_path,  function="f_01", initial_guess=[0,0], niter=100, T=100, stepsize=0.1,  image_name = "Default", save_image=False, seed = 42):
        #Core function implementing the search for a minimum in two functions and comparing them, one neural net, one ground-truth
        self.model = self.load_model(model_path) #Load neural net
        self.model.eval()
        self.save_image=save_image
        
        #Load ground-Truth Function
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
        
        #Search through approximation for minimum
        result_nn = basinhopping(self.call_nn, initial_guess, niter=niter, T=T, stepsize=stepsize, seed=seed, callback=self.callback)
        model_path = np.array(self.path)
        self.path= []
        
        #Search through Ground-truth function for minimum
        result_bbob = basinhopping(self.call_bbob, initial_guess, niter=niter, T=T, stepsize=stepsize, seed=seed, callback=self.callback)
        bbob_path = np.array(self.path)

        if save_image:
            fig = self.visualize_paths(bbob_path, model_path, result_bbob, result_nn) #visualize paths taken if save_image

        return result_nn, result_bbob, model_path, bbob_path
    
    def visualize_paths(self, path_bbob, path_model, result_bbob, result_model):
        #Visualization of all points visisted during search within both bbob and nn functions
        x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))
        
        # Convert the scaled grid data to PyTorch tensor
        grid_data_tensor = torch.tensor(np.column_stack((x_grid.ravel(), y_grid.ravel())), dtype=torch.float32)

        # Set the model to evaluation mode
        self.model.eval()

        # Pass the grid data through the model to get predictions
        with torch.no_grad():
            predictions = self.model(grid_data_tensor).numpy().reshape(x_grid.shape)
            
        function_values = self._get_gt_function(x_grid, y_grid)
        
        fig, axes = plt.subplots(1, 2, figsize= (16,8))
        
        for ax, model_values, path, result in zip(axes, [predictions, function_values], [path_model, path_bbob], [result_model, result_bbob]):
            
            ax.contourf(x_grid, y_grid, model_values, levels=100, cmap='viridis')

            ax.plot(path[:,0], path[:,1], '-ro', markersize=1, linewidth=0.5)
            
            ax.scatter(x= result['x'][0], y= result['x'][1], c='black', s= 150, marker='x')
            
    
        if self.save_image:
            # Save the plot as a PNG file
            plt.savefig(f'images/basinhopping/{self.image_name}', dpi=300)
        #plt.show()
        return plt
    

    def callback(self, x, f, accept):
        #Callback to save points visited
        self.path.append(x)
    
    def _get_gt_function(self, x_grid, y_grid):
        #Get function landscape for visualization
        fn = self.bbob
        flat_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Convert flat_grid to a PyTorch Tensor
        flat_grid_tensor = torch.tensor(flat_grid, dtype=torch.float32)

        # Evaluate the ground-truth function using the bbobtorch function
        results = fn(flat_grid_tensor)

        return results.numpy().reshape(x_grid.shape) 

        