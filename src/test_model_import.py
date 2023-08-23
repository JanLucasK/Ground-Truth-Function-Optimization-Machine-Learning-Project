import json
import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np
import pandas as pd
import bbobtorch
import matplotlib.pyplot as plt
import seaborn as sns



def plot_heatmap(x_grid, y_grid, predictions):
    


    sns.heatmap(predictions, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.show()




def evaluate_grid(model):
    # Generate the grid of input data
    x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))

    # Scale the grid data using the same StandardScaler
    # grid_data_scaled = self.scaler.transform(np.column_stack((x_grid.ravel(), y_grid.ravel())))
    grid_data = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Convert the scaled grid data to PyTorch tensor
    grid_data_tensor = torch.tensor(grid_data, dtype=torch.float32)


    # Pass the grid data through the model to get predictions
    with torch.no_grad():
        predictions = model(grid_data_tensor).view(x_grid.shape)

    # Plot the heatmaps
    plot_heatmap(x_grid, y_grid, predictions)

def import_model():


    current_directory = os.getcwd()
    print("Current working directory:", current_directory)

    model = torch.load("models/test.pth")
    model.eval()
    return model



evaluate_grid(import_model())
