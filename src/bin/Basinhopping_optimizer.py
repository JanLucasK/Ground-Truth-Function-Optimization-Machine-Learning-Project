import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import bbobtorch
from scipy.optimize import basinhopping
from bin.Visualization import visualization as vis

class Basinhopping_optimizer():
    def __init__(self, model_path, input_bounds=None) -> None:   
        self.model = self.load_model(model_path)
        self.input_bounds = input_bounds
        self.model.eval()
        self.bbob = bbobtorch.create_f24(2, seed=42)
   
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
        return y_scalar
    
    def optimize(self, initial_guess, niter=1000, T=1.0, stepsize=0.05):
        result_nn = basinhopping(self.call_nn, initial_guess, niter=niter, T=T, stepsize=stepsize)
        result_bbob = basinhopping(self.call_bbob, initial_guess, niter=niter, T=T, stepsize=stepsize)
        #self.visualize(result_nn, result_bbob)
        return result_nn, result_bbob
    
    def visualize(self, result_nn, result_bbob):
        heatmap = vis(model= self.model, function=self.bbob)
        heatmap.evaluate_grid(nn_results=result_nn, bbob_results=result_bbob)
    
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
        return y_scalar
    
   